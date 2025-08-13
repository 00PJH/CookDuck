from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from accelerate import Accelerator
import torch
import shutil
import pandas as pd
import os
import gc
import threading
import time

# 0. GPU 설정
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("✅ GPU 설정 완료")
except Exception as e:
    print("❌ GPU 설정 오류:", e)
    exit()

# VRAM 사용량 체크 함수 추가
def print_vram_usage():
    print("\n[VRAM 사용량 체크]")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} / 사용량: {round(torch.cuda.memory_allocated(i)/1024**3,2)} GB / 총: {round(torch.cuda.get_device_properties(i).total_memory/1024**3,2)} GB")

# VRAM 자동 모니터링 스레드 시작
def monitor_vram(interval_sec=300):
    while True:
        print_vram_usage()
        time.sleep(interval_sec)

monitor_thread = threading.Thread(target=monitor_vram, daemon=True)
monitor_thread.start()

# 학습 로그 저장 설정
log_file = open("training_log_1b.txt", "w")
def log_print(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

print_vram_usage()

# 1. 모델 및 토크나이저 로딩
try:
    log_print("\n## 1. 모델 로딩 시작")
    model_id = "torchtorchkimtorch/Llama-3.2-Korean-GGACHI-1B-Instruct-v1"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    log_print("✅ 모델 로딩 완료")
    print_vram_usage()
except Exception as e:
    log_print("❌ 모델 로딩 오류: " + str(e))
    exit()

# 2. 데이터 로딩 및 전처리 
try:
    log_print("\n## 2. 데이터 로딩 시작")
    data_path = "translation_korean_to_english_data/en_to_ko_shuffle_small_30k.csv"
    processed_data_path = "translation_korean_to_english_data/processed_dataset"

    if os.path.exists(processed_data_path):
        log_print("✅ 기존 processed_dataset 삭제 중...")
        shutil.rmtree(processed_data_path)

    df = pd.read_csv(data_path)

    # ✅ 컬럼 이름 강제 설정 수정 (english, korean만 존재)
    df.columns = ['english', 'korean']

    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(processed_data_path)
    log_print("✅ 데이터 저장 완료 (english, korean 포함)")

    log_print("✅ 데이터 로딩 완료. 샘플:")
    log_print(str(dataset[0]))
except Exception as e:
    log_print("❌ 데이터 로딩 오류: " + str(e))
    exit()

# 3. 모델 준비 (qLoRA 적용)
try:
    log_print("\n## 3. qLoRA 세팅 시작")
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    log_print("✅ qLoRA 설정 완료")
    print_vram_usage()
except Exception as e:
    log_print("❌ qLoRA 설정 오류: " + str(e))
    exit()


# 4. 전처리 및 DataLoader 준비 (이제 여기서 토크나이징)
try:
    log_print("\n## 4. DataLoader 준비 및 전처리 시작")

    def generate_prompt(data_point):
        return f"### Instruction:\nTranslate the following English sentence to Korean.\n\n### Input:\n{data_point['english']}\n\n### Response:\n{data_point['korean']}"

    def tokenize_fn(data_point):
        prompt = generate_prompt(data_point)
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'][0],
            'attention_mask': tokenized['attention_mask'][0]
        }

    dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(example['input_ids']) for example in batch])
        attention_mask = torch.stack([torch.tensor(example['attention_mask']) for example in batch])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn)

    log_print("✅ DataLoader 및 전처리 완료")
except Exception as e:
    log_print("❌ DataLoader 준비 오류: " + str(e))
    exit()



# 가장 최근 체크포인트 찾기 함수
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-epoch-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

# 5. 학습 시작 (Custom Train Loop + Checkpoint 저장)
try:
    log_print("\n## 6. Custom 학습 시작")

    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        log_print(f"✅ 최신 체크포인트 복원: {latest_checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            latest_checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3*len(train_dataloader))

    model.train()
    num_epochs = 4
    save_path = "../../finetuned_models/Llama-3.2-Korean-GGACHI-1B-Instruct-v1-koToEn"
    best_val_loss = float('inf')

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    start_epoch = 0
    if latest_checkpoint:
        start_epoch = int(latest_checkpoint.split("-")[-1]) + 1

    for epoch in range(start_epoch, num_epochs):
        start_time_epoch = time.time()  # Epoch 시작 시간 측정
        total_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training Epoch {epoch}", ncols=100):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                log_print(f"\nEpoch {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_train_loss = total_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)
        log_print(f"Epoch {epoch} Finished. Avg Train Loss: {avg_train_loss:.4f}")

        epoch_duration = time.time() - start_time_epoch  # Epoch 시간 계산
        log_print(f"Epoch {epoch} Duration: {epoch_duration:.2f} seconds")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            # for batch in val_dataloader:
            #     batch = {k: v.to(model.device) for k, v in batch.items()}
            #     outputs = model(**batch)
            #     loss = outputs.loss
            #     val_loss += loss.item()

            #     predictions = outputs.logits.argmax(dim=-1)
            #     correct += (predictions == batch['labels']).float().sum().item()
            #     total += batch['labels'].numel()
            for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch}", ncols=100):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()

                predictions = outputs.logits.argmax(dim=-1)
                correct += (predictions == batch['labels']).float().sum().item()
                total += batch['labels'].numel()


        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total
        val_loss_list.append(avg_val_loss)
        val_accuracy_list.append(val_accuracy)

        log_print(f"\nEpoch {epoch} Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        # Best 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            log_print(f"Saving best model at epoch {epoch} with validation loss {avg_val_loss:.4f}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        # 매 epoch마다 checkpoint 저장
        model.save_pretrained(os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}"))
        tokenizer.save_pretrained(os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch}"))

    log_print("✅ 학습 완료")

    # Save loss and accuracy history
    history_df = pd.DataFrame({
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'val_accuracy': val_accuracy_list
    })
    history_df.to_csv("loss_accuracy_history.csv", index=False)

    print_vram_usage()
except Exception as e:
    log_print("❌ 학습 오류: " + str(e))
    exit()

# 로그 파일 닫기
log_file.close()