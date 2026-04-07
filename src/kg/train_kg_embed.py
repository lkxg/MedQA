from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch
import torch.nn.functional as F
import json
import time
import os

# ⭐ 固定随机种子，保证实验可复现
torch.manual_seed(42)

# 检查GPU
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = 'cuda'
else:
    print("没有GPU，使用CPU")
    device = 'cpu'

# 加载数据
tf = TriplesFactory.from_path(
    '../../data/medical_kg_data/triples.tsv',
    create_inverse_triples=True,    # ⭐ 医疗关系建模双向：(糖尿病→多饮) + (多饮→糖尿病)
)
training, testing, validation = tf.split(
    [0.8, 0.1, 0.1],
    random_state=42,                # ⭐ 固定划分，实验可复现
)
print(f"实体: {tf.num_entities}, 关系: {tf.num_relations} (含逆关系)")
print(f"训练: {training.num_triples}, 测试: {testing.num_triples}, 验证: {validation.num_triples}")

# 训练
print(f"\n在 {device} 上训练...")
t_start = time.time()

result = pipeline(
    model='RotatE',
    training=training,
    testing=testing,
    validation=validation,
    
    device=device,
    random_seed=42,                      # ⭐ 模型初始化种子
    stopper='early',
    stopper_kwargs=dict(frequency=5, patience=10, relative_delta=0.0005),
    
    model_kwargs=dict(
        embedding_dim=256,
    ),
    training_kwargs=dict(
        num_epochs=500,
        batch_size=2048,
    ),
    optimizer='Adam',
    optimizer_kwargs=dict(
        lr=5e-4,                         # ⭐ 降低学习率，更稳收敛 (原 1e-3 偏高)
    ),
    lr_scheduler='CosineAnnealingWarmRestarts',  # ⭐ 余弦退火，比 StepLR 更平滑
    lr_scheduler_kwargs=dict(
        T_0=50,                          # 首个周期 50 epochs
        T_mult=2,                        # 每次周期翻倍 (50→100→200)
        eta_min=1e-6,                    # 最低学习率
    ),
    loss='nssa',                         # RotatE 论文标配：自对抗负采样损失
    loss_kwargs=dict(
        margin=9.0,
        adversarial_temperature=1.0,
    ),
    negative_sampler='basic',
    negative_sampler_kwargs=dict(
        num_negs_per_pos=64,
    ),
    regularizer='lp',
    regularizer_kwargs=dict(
        weight=1e-5,
        p=3.0,
    ),
    evaluation_kwargs=dict(
        batch_size=1024,
    ),
)

t_elapsed = time.time() - t_start
print(f"\n⏱️ 训练耗时: {t_elapsed/60:.1f} 分钟")

# 获取并打印专业的链接预测指标 (Link Prediction Metrics)
print("\n" + "="*50)
print("📊 终局测试集: 链接预测 (Link Prediction) 核心指标")
print("="*50)
mrr = result.metric_results.get_metric('mean_reciprocal_rank')
hits_at_1 = result.metric_results.get_metric('hits@1')
hits_at_3 = result.metric_results.get_metric('hits@3')
hits_at_10 = result.metric_results.get_metric('hits@10')
print(f"MRR      : {mrr:.4f}  (平均倒数排名)")
print(f"Hits@1   : {hits_at_1:.4f}  (Top-1 命中率)")
print(f"Hits@3   : {hits_at_3:.4f}  (Top-3 命中率)")
print(f"Hits@10  : {hits_at_10:.4f} (Top-10 命中率)")
print("="*50 + "\n")

# 提取嵌入（移到CPU保存）
entity_to_id = tf.entity_to_id
id_to_entity = {v: k for k, v in entity_to_id.items()}

entity_emb = result.model.entity_representations[0](
    torch.arange(len(entity_to_id)).to(device)    # ⭐ 放到GPU上
).detach().cpu()                                    # ⭐ 取回CPU

relation_emb = result.model.relation_representations[0](
    torch.arange(len(tf.relation_to_id)).to(device)
).detach().cpu()

print(f"实体嵌入: {entity_emb.shape}")

# 验证
entity_emb_real = entity_emb.real if entity_emb.is_complex() else entity_emb
entity_emb_normed = F.normalize(entity_emb_real.float(), p=2, dim=1)

triples_set = set()
with open('../../data/medical_kg_data/triples.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            triples_set.add((parts[0], parts[2]))
            triples_set.add((parts[2], parts[0]))

for name in ['糖尿病', '高血压', '感冒', '头痛', '阿司匹林']:
    if name not in entity_to_id:
        continue
    idx = entity_to_id[name]
    query = entity_emb_normed[idx].unsqueeze(0)
    sims = torch.mm(query, entity_emb_normed.T).squeeze()
    topk_vals, topk_idxs = torch.topk(sims, k=6)
    
    print(f"\n与 [{name}] 最相似:")
    for v, i in zip(topk_vals[1:], topk_idxs[1:]):
        n = id_to_entity[i.item()]
        mark = "✅" if (name, n) in triples_set else "  "
        print(f"  {mark} {n}: {v.item():.4f}")

# 保存嵌入
torch.save(entity_emb_real.float(), '../../kg_embedding/entity_embeddings.pt')

relation_emb_real = relation_emb.real if relation_emb.is_complex() else relation_emb
torch.save(relation_emb_real.float(), '../../kg_embedding/relation_embeddings.pt')

with open('../../kg_embedding/entity_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(dict(entity_to_id), f, ensure_ascii=False, indent=2)
with open('../../kg_embedding/relation_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(dict(tf.relation_to_id), f, ensure_ascii=False, indent=2)

# ⭐ 保存完整模型 checkpoint（避免重复训练）
checkpoint_dir = '../../kg_embedding/rotate_checkpoint'
result.save_to_directory(checkpoint_dir)
print(f"💾 模型 checkpoint 已保存至: {checkpoint_dir}")

# ⭐ 保存训练指标到 JSON（方便对比实验）
metrics = {
    'MRR': mrr, 'Hits@1': hits_at_1, 'Hits@3': hits_at_3, 'Hits@10': hits_at_10,
    'training_time_min': round(t_elapsed / 60, 1),
    'num_entities': len(entity_to_id),
    'num_relations': len(tf.relation_to_id),
    'embedding_dim': 256,
    'inverse_triples': True,
}
with open('../../kg_embedding/training_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("\n✅ 全部完成！")
print(f"   嵌入: ../../kg_embedding/entity_embeddings.pt ({entity_emb_real.shape})")
print(f"   模型: {checkpoint_dir}/")
print(f"   指标: training_metrics.json")