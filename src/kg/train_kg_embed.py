import json
import os
import time

import torch
import torch.nn.functional as F
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# ── 配置──
MODEL_NAME = 'RotatE'   # 'RotatE' or 'ComplEx'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '../../data/medical_kg_data')
EMB_DIR    = os.path.join(SCRIPT_DIR, f'../../kg_embedding/{MODEL_NAME.lower()}')

torch.manual_seed(42)

# ── 环境检查 ─────────────────────────────────────────────────────────
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = 'cuda'
else:
    print("没有GPU，使用CPU")
    device = 'cpu'

os.makedirs(EMB_DIR, exist_ok=True)

# ── 数据加载 ─────────────────────────────────────────────────────────
training = TriplesFactory.from_path(
    f'{DATA_DIR}/train.tsv',
    create_inverse_triples=True,
)
validation = TriplesFactory.from_path(
    f'{DATA_DIR}/valid.tsv',
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
    create_inverse_triples=True,
)
testing = TriplesFactory.from_path(
    f'{DATA_DIR}/test.tsv',
    entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,
    create_inverse_triples=True,
)
tf = training
print(f"实体: {tf.num_entities}, 关系: {tf.num_relations} (含逆关系)")
print(f"训练: {training.num_triples}, 验证: {validation.num_triples}, 测试: {testing.num_triples}")

# ── 模型参数（按MODEL_NAME自动切换）─────────────────────────────────
if MODEL_NAME == 'RotatE':
    model_cfg = dict(
        loss='nssa',
        loss_kwargs=dict(margin=12.0, adversarial_temperature=1.0),
    )
    opt_kwargs = dict(lr=5e-4)
    reg_kwargs = dict(weight=1e-5, p=3.0)
elif MODEL_NAME == 'ComplEx':
    model_cfg = dict(
        loss='softplus',
        loss_kwargs=dict(),
    )
    opt_kwargs = dict(lr=2e-4)
    reg_kwargs = dict(weight=1e-6, p=2.0)
else:
    raise ValueError(f"不支持的模型: {MODEL_NAME}，请选择 RotatE 或 ComplEx")

# ── 训练 ─────────────────────────────────────────────────────────────
print(f"\n使用 {MODEL_NAME} 在 {device} 上训练...")
t_start = time.time()

result = pipeline(
    model=MODEL_NAME,
    training=training,
    testing=testing,
    validation=validation,
    device=device,
    random_seed=42,

    model_kwargs=dict(embedding_dim=512),

    **model_cfg,    # loss / loss_kwargs 按模型自动注入

    training_kwargs=dict(num_epochs=500, batch_size=2048),

    optimizer='Adam',
    optimizer_kwargs=opt_kwargs,

    lr_scheduler='CosineAnnealingWarmRestarts',
    lr_scheduler_kwargs=dict(T_0=50, T_mult=2, eta_min=1e-6),

    stopper='early',
    stopper_kwargs=dict(frequency=5, patience=10, relative_delta=0.0005),

    negative_sampler='basic',
    negative_sampler_kwargs=dict(num_negs_per_pos=64),

    regularizer='lp',
    regularizer_kwargs=reg_kwargs,

    evaluation_kwargs=dict(batch_size=1024),
)

t_elapsed = time.time() - t_start
print(f"\n⏱️  训练耗时: {t_elapsed / 60:.1f} 分钟")

# ── 评估指标 ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"📊 {MODEL_NAME} 测试集链接预测指标")
print("=" * 50)
mrr        = result.metric_results.get_metric('mean_reciprocal_rank')
hits_at_1  = result.metric_results.get_metric('hits@1')
hits_at_3  = result.metric_results.get_metric('hits@3')
hits_at_10 = result.metric_results.get_metric('hits@10')
print(f"MRR    : {mrr:.4f}")
print(f"Hits@1 : {hits_at_1:.4f}")
print(f"Hits@3 : {hits_at_3:.4f}")
print(f"Hits@10: {hits_at_10:.4f}")
print("=" * 50)

# ── 提取嵌入 ─────────────────────────────────────────────────────────
def to_real(emb):
    """复数嵌入→实数：拼接实部+虚部，保留完整表示"""
    if emb.is_complex():
        return torch.cat([emb.real, emb.imag], dim=-1).float()
    return emb.float()

entity_to_id = tf.entity_to_id
id_to_entity = {v: k for k, v in entity_to_id.items()}

entity_emb = result.model.entity_representations[0](
    torch.arange(len(entity_to_id)).to(device)
).detach().cpu()

relation_emb = result.model.relation_representations[0](
    torch.arange(len(tf.relation_to_id)).to(device)
).detach().cpu()

entity_emb_real   = to_real(entity_emb)    # (N, 512)
relation_emb_real = to_real(relation_emb)
print(f"\n实体嵌入维度: {entity_emb_real.shape}")

# ── 语义近邻验证 ─────────────────────────────────────────────────────
entity_emb_normed = F.normalize(entity_emb_real, p=2, dim=1)

triples_set = set()
with open(f'{DATA_DIR}/triples.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            triples_set.add((parts[0], parts[2]))
            triples_set.add((parts[2], parts[0]))

for name in ['糖尿病', '高血压', '感冒', '心肌梗死', '阿司匹林']:
    if name not in entity_to_id:
        continue
    idx   = entity_to_id[name]
    query = entity_emb_normed[idx].unsqueeze(0)
    sims  = torch.mm(query, entity_emb_normed.T).squeeze()
    topk_vals, topk_idxs = torch.topk(sims, k=6)
    print(f"\n与 [{name}] 最相似:")
    for v, i in zip(topk_vals[1:], topk_idxs[1:]):
        n    = id_to_entity[i.item()]
        mark = "✅" if (name, n) in triples_set else "  "
        print(f"  {mark} {n}: {v.item():.4f}")

# ── 保存 ─────────────────────────────────────────────────────────────
torch.save(entity_emb_real,   f'{EMB_DIR}/entity_embeddings.pt')
torch.save(relation_emb_real, f'{EMB_DIR}/relation_embeddings.pt')

with open(f'{EMB_DIR}/entity_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(dict(entity_to_id), f, ensure_ascii=False, indent=2)
with open(f'{EMB_DIR}/relation_to_id.json', 'w', encoding='utf-8') as f:
    json.dump(dict(tf.relation_to_id), f, ensure_ascii=False, indent=2)

checkpoint_dir = f'{EMB_DIR}/checkpoint'
result.save_to_directory(checkpoint_dir)

metrics = {
    'model':              MODEL_NAME,
    'MRR':                mrr,
    'Hits@1':             hits_at_1,
    'Hits@3':             hits_at_3,
    'Hits@10':            hits_at_10,
    'training_time_min':  round(t_elapsed / 60, 1),
    'num_entities':       len(entity_to_id),
    'num_relations':      len(tf.relation_to_id),
    'embedding_dim':      256,
    'actual_emb_dim':     entity_emb_real.shape[1],  # 512 (256*2)
    'inverse_triples':    True,
    'note':               'relation_to_id含逆关系，原始关系数为num_relations//2',
}
with open(f'{EMB_DIR}/training_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n✅ {MODEL_NAME} 训练完成！")
print(f"   实体嵌入: {EMB_DIR}/entity_embeddings.pt  {entity_emb_real.shape}")
print(f"   模型:     {checkpoint_dir}/")
print(f"   指标:     {EMB_DIR}/training_metrics.json")