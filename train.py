import random, functools, torch.distributed as dist, wandb, uuid, torch, transformers, os, math, numpy as np

import time
import bitsandbytes as bnb
from datasets import load_dataset, DatasetDict, Dataset
from functools import partial
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datetime import datetime
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
    CPUOffload
)

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from collections import namedtuple
from torch.nn import CrossEntropyLoss
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Mamba, Block
from pynvml import *


def print_gpu_utilization(rank):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU{rank} memory occupied: {info.used//1024**2} MB.")

def train():
    local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    # 100 Parameters
    wandb_log=True

    model_name = "state-spaces/mamba-130m"
    scheduler_type = "constant"
    transformers.set_seed(42)

    run_id = str(uuid.uuid4())
    output_dir = f"./outputs/{model_name}/{run_id}"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 8_000  
    disable_dropout = False
    gradient_checkpointing = False
    clip_gradients = True
    shuffle = True  
    train_batch_size, validation_batch_size = 2, 1
    epochs = 3  
    lr = 2e-05
    weight_decay = 0.0  
    gradient_clipping = 1.0  

    # Load model and tokenizer    
    print("Loading model and tokenizer")
    model, tokenizer = setup_model(model_name)
    print("done")

    # Wrap model like RWKV: https://github.com/mrsteyk/RWKV-LM-deepspeed/blob/89a632ff61094b4d6629d4daa379b27e296d3400/RWKV-v4f/train.py#L208
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block
        },
    )
    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=None,
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=CPUOffload(offload_params=False),
    )
    print("Wrapping model")
    model = FSDP(model, **fsdp_config)
    print("done")

    optimizer = get_optimizer(model, lr, weight_decay)

    # load dataset
    dataset_name="wikimedia/wikisource"
    dataset=load_dataset(dataset_name, name="20231201.en")
    # dataset["train"]=dataset["train"].select(range(0, 1_000))
    dataset=dataset["train"].train_test_split(test_size=0.05)   

    # tokenize dataset
    dataset_tokenized = dataset.map(
        partial(tokenize, max_length=max_length, tokenizer=tokenizer), 
        batched=True, 
        num_proc=os.cpu_count()//world_size,    # multithreaded
        remove_columns=["text"]     # don't need this anymore, we have tokens from here on
    )

    # Prepare data sampler and loader
    my_get_dataloader=partial(
            get_dataloader,
            collator=partial(collate, tokenizer=tokenizer),
            fsdp_info=[local_rank, world_size]
        )
    train_sampler, train_loader = my_get_dataloader(dataset=dataset_tokenized["train"], bs=train_batch_size)
    val_sampler, val_loader = my_get_dataloader(dataset=dataset_tokenized["test"], bs=validation_batch_size)

    total_steps_per_epoch = len(train_loader)
    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        print(model)

        if wandb_log:
            run = wandb.init(
                project="mamba",
                name=model_name.split("/")[1]+"_"+dataset_name.split("/")[1]+f"_bs-{train_batch_size}_LR-{lr}_maxlen-{max_length}_{run_id}",
                config={
                    "model_name": model_name,
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "output_dir": output_dir,
                    "lr": lr,
                    "max_length": max_length,
                    "train_batch_size": train_batch_size,
                    "validation_batch_size": validation_batch_size
                }
            )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    torch.cuda.empty_cache()
    model.train()
    dist.barrier()

    print_gpu_utilization(local_rank)    
    time.sleep(10)

    token_count=0    # tokens trained
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_loader),
            total=total_steps_per_epoch,
            colour="blue",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )

        for step, batch in pbar:
            current_step = step + 1

            inputs = {
                "input_ids": batch["input_ids"].to("cuda"),
                "labels": batch["labels"].to("cuda"),
            }
            token_count+=batch["token_count"]

            # forward
            outputs = model(**inputs)
            loss = outputs.loss

            # backward
            loss.backward()

            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            # log every 4 steps
            if current_step % 4 == 0:
                token_count_gathered = gather_object( [token_count] )
                if local_rank == 0:
                    log_stats(
                        pbar,
                        wandb,
                        round((current_step / total_steps_per_epoch), 2) + epoch,
                        loss,
                        grad_norm,
                        scheduler,
                        sum(token_count_gathered)
                    )

            # eval 5 times per epoch
            if should_run_eval(total_steps_per_epoch, 5, current_step):
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                )
                save_model(local_rank, model, tokenizer, output_dir, current_epoch,current_step)
                model.train()

    save_model(local_rank, model, tokenizer, output_dir, epochs, "final")


def collate(elements, tokenizer):
    tokenlist=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokenlist])
    token_count=0

    input_ids,labels,attention_masks = [],[],[]
    for tokens in tokenlist:
        pad_len=tokens_maxlen-len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
        input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
        labels.append( tokens + [-100]*pad_len )    
        attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

        token_count+=len(tokens)

    batch={
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "token_count": token_count
    }
    return batch


def gather_object(object):
    output_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output_objects, object)
    return [x for y in output_objects for x in y]


def tokenize(element, max_length, tokenizer):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

def get_dataloader(
    dataset,
    bs,
    collator,
    fsdp_info,
    shuffle=False,
    seed=42,
):
    fsdp_rank, fsdp_world_size = fsdp_info

    sampler = DistributedSampler(dataset=dataset, rank=fsdp_rank, num_replicas=fsdp_world_size, shuffle=shuffle)
    loader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        pin_memory=True,
        batch_size=bs,
        collate_fn=collator,
        sampler=sampler,
    )
    return sampler, loader


def disable_model_dropout(model: torch.nn.Module):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def setup_model(model_name):
    # monkey patch MambaLMHeadModel.forward 
    def forward_with_loss(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, labels = None):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        
        # Source: https://github.com/huggingface/transformers/blob/80377eb018c077dba434bc8e7912bcaed3a64d09/src/transformers/models/llama/modeling_llama.py#L1196
        if labels is not None:
            logits = lm_logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_logits = shift_logits.view(-1, self.backbone.embedding.weight.size()[0])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            CausalLMOutput = namedtuple("CausalLMOutput", ["loss"])
            return CausalLMOutput(loss=loss)            
            # return (loss,)   
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
            return CausalLMOutput(logits=lm_logits)
    MambaLMHeadModel.forward=forward_with_loss

    model = MambaLMHeadModel.from_pretrained(
        model_name,    
    )

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") 
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def evaluation(
    model,
    eval_dataloader,
    wandb,
    local_rank,
):
    if local_rank == 0: print("RUNNING EVAL")

    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        inputs = {
            "input_ids": batch["input_ids"].to("cuda"),
            "labels": batch["labels"].to("cuda"),
        }
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    val_loss = get_all_reduce_mean(losses.clone()).item()

    if local_rank == 0: wandb.log({"val_loss": val_loss})

    return val_loss



def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]

    result += list(model._parameters.keys())
    return result


def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    # https://www.kaggle.com/code/nbroad/8-bit-adam-optimization
    # optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=lr)
    # for module in model.modules():
    #     if isinstance(module, torch.nn.Embedding):
    #         bnb.optim.GlobalOptimManager.get_instance().register_module_override(
    #             module, 'weight', {'optim_bits': 32}
    #         )     
    # return optimizer
    # https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one
    # adam_bnb_optim = bnb.optim.Adam8bit(
    #     params=optimizer_grouped_parameters,
    #     # betas=(0.9, 0.95),
    #     # eps=training_args.adam_epsilon,
    #     lr=lr
    # )    

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def should_run_eval(total_steps, times_to_run, current_step):
    return current_step % (total_steps // times_to_run) == 0


def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler, token_count):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
            "token_count": token_count, 
        },
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_warmup_steps(num_training_steps, warmup_ratio=0.05):
    return math.ceil(num_training_steps * warmup_ratio)

def get_scheduler(local_rank, scheduler_type, optimizer, max_steps):
    warmup_steps = get_warmup_steps(max_steps)

    return transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

def clip_model_gradients(model, max_grad_norm):
    return model.clip_grad_norm_(max_grad_norm).item()

def save_model(local_rank, model, tokenizer, outpath, current_epoch, current_step):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        outpath += f"/epoch_{current_epoch}/step_{current_step}"
        os.makedirs(outpath)
        # model.save_pretrained(outpath, state_dict=cpu_state)
        torch.save(cpu_state, f"{outpath}/pytorch_model.bin")
        tokenizer.save_pretrained(outpath)


if __name__ == "__main__":
    train()