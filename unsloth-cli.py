#!/usr/bin/env python3

"""
🦥 Unsloth ile FastLanguageModel'i ince ayar yapmak için başlangıç betiği

Bu betik, modellerinizi unsloth kullanarak ince ayar yapmak için başlangıç noktası olarak tasarlanmıştır.
Model yükleme, PEFT parametreleri, eğitim argümanları ve modeli kaydetme/gönderme işlevleri gibi yapılandırılabilir seçenekler içerir.

Kendi kullanım senaryonuza ve gereksinimlerinize uygun olacak şekilde bu betiği özelleştirmek isteyebilirsiniz.

Özelleştirme için birkaç öneri:
    - Veri kümesi yükleme ve ön işleme adımlarını verilerinize uyacak şekilde değiştirin.
    - Modeli kaydetme ve gönderme yapılandırmalarını kişiselleştirin.

Kullanım: (seçeneklerin çoğu geçerli varsayılan değerlere sahiptir, bu uzun bir örnektir)
    python unsloth-cli.py --model_name "unsloth/llama-3-8b" --max_seq_length 8192 --dtype None --load_in_4bit \
    --r 64 --lora_alpha 32 --lora_dropout 0.1 --bias "none" --use_gradient_checkpointing "unsloth" \
    --random_state 3407 --use_rslora --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --warmup_steps 5 --max_steps 400 --learning_rate 2e-6 --logging_steps 1 --optim "adamw_8bit" \
    --weight_decay 0.005 --lr_scheduler_type "linear" --seed 3407 --output_dir "outputs" \
    --report_to "tensorboard" --save_model --save_path "model" --quantization_method "f16" \
    --push_model --hub_path "hf/model" --hub_token "your_hf_token"

Tüm yapılandırılabilir seçenekleri görmek için:
    python unsloth-cli.py --help

İyi ince ayarlar!
"""

import argparse
import os


def run(args):
    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from transformers.utils import strtobool
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import logging
    logging.getLogger('hf-to-gguf').setLevel(logging.WARNING)

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    use_modelscope = strtobool(os.environ.get('UNSLOTH_USE_MODELSCOPE', 'False'))
    if use_modelscope:
        from modelscope import MsDataset
        dataset = MsDataset.load(args.dataset, split="train")
    else:
        # Load and format dataset
        dataset = load_dataset(args.dataset, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print("Veriler biçimlendirildi ve hazır!")

    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train model
    trainer_stats = trainer.train()

    # Save model
    if args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    print(f"Model {quantization_method} ile miktarlandırılarak kaydediliyor")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                print(f"Model {args.quantization} ile miktarlandırılarak kaydediliyor")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=quantization_method,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        print("Uyarı: Model kaydedilmedi!")


if __name__ == "__main__":

    # Define argument parser
    parser = argparse.ArgumentParser(description="🦥 Unsloth ile LLM'inizi daha hızlı ince ayar yapın!")

    model_group = parser.add_argument_group("🤖 Model Seçenekleri")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Yüklenecek model adı")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maksimum dizi uzunluğu, varsayılan 2048. Dahili olarak RoPE Ölçeklendirmesini otomatik destekliyoruz!")
    model_group.add_argument('--dtype', type=str, default=None, help="Model için veri tipi (otomatik algılama için None)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Bellek kullanımını azaltmak için 4 bit miktarlandırma kullan")
    model_group.add_argument('--dataset', type=str, default="yahma/alpaca-cleaned", help="Eğitim için kullanılacak Huggingface veri kümesi")

    lora_group = parser.add_argument_group("🧠 LoRA Seçenekleri", "LoRA modelini yapılandırmak için kullanılır.")
    lora_group.add_argument('--r', type=int, default=16, help="LoRA modeli için derece, varsayılan 16 (yaygın değerler: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parametresi, varsayılan 16 (yaygın değerler: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout oranı, varsayılan 0.0 olup optimize edilmiştir")
    lora_group.add_argument('--bias', type=str, default="none", help="LoRA için bias ayarı")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Gradient checkpointing kullan")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Tekrarlanabilirlik için rastgele durum, varsayılan 3407")
    lora_group.add_argument('--use_rslora', action='store_true', help="Sıra stabilize LoRA kullan")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="LoftQ yapılandırması")

   
    training_group = parser.add_argument_group("🎓 Eğitim Seçenekleri")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Eğitim sırasında cihaz başına batch boyutu, varsayılan 2")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Gradient biriktirme adımları, varsayılan 4")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Isınma adımları sayısı, varsayılan 5")
    training_group.add_argument('--max_steps', type=int, default=400, help="Eğitim adımlarının maksimum sayısı")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Öğrenme oranı, varsayılan 2e-4")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizasyon türü")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Ağırlık azalması, varsayılan 0.01")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Öğrenme oranı zamanlayıcı türü, varsayılan 'linear'")
    training_group.add_argument('--seed', type=int, default=3407, help="Tekrarlanabilirlik için tohum, varsayılan 3407")
    

    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Raporlama Seçenekleri")
    report_group.add_argument('--report_to', type=str, default="tensorboard",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="Sonuç ve günlükleri göndereceğiniz entegrasyonların listesi. Desteklenen platformlar: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard' ve 'wandb'. Tüm kurulu entegrasyonlara raporlamak için 'all', hiçbiri için 'none' kullanın.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Günlükleme adımları, varsayılan 1")

    # Saving and pushing arguments
    save_group = parser.add_argument_group('💾 Model Kaydetme Seçenekleri')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Çıktı dizini")
    save_group.add_argument('--save_model', action='store_true', help="Eğitimden sonra modeli kaydet")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Modeli kaydetme yöntemi, varsayılan 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Eğitimden sonra modeli GGUF'e dönüştür")
    save_group.add_argument('--save_path', type=str, default="model", help="Modelin kaydedileceği yol")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Modeli kaydederken kullanılacak miktarlandırma yöntemi. yaygın değerler ('f16', 'q4_k_m', 'q8_0'); tüm yöntemler için wiki sayfamıza bakın https://github.com/unslothai/unsloth/wiki#saving-to-gguf")

    push_group = parser.add_argument_group('🚀 Modeli Yükleme Seçenekleri')
    push_group.add_argument('--push_model', action='store_true', help="Eğitimden sonra modeli Hugging Face hub'a yükle")
    push_group.add_argument('--push_gguf', action='store_true', help="Modeli GGUF olarak Hugging Face hub'a yükle")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Hub'da modeli yüklemek için hedef yol")
    push_group.add_argument('--hub_token', type=str, help="Hugging Face hub'a yüklemek için gerekli token")

    args = parser.parse_args()
    run(args)
