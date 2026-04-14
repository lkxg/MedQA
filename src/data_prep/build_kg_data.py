import json
import os
import random
from collections import defaultdict


class MedicalKGBuilder:
    def __init__(self):
        self.triples = set()
        self.entities = set()
        self.relations = set()
        self.entity_types = {}
        self.disease_desc = {}  # 存储疾病描述文本，供Graph-CoT使用

        # 核心7条关系
        self.mapping = {
            'symptom':        ('有症状',   '症状'),
            'check':          ('需检查',   '检查'),
            'acompany':       ('并发症为', '疾病'),
            'recommand_drug': ('推荐药物', '药物'),
            'cure_way':       ('治疗方式', '治疗'),
            'cure_department':('所属科室', '科室'),
            'easy_get':       ('易感人群', '人群'),
        }

    def add_from_qa_system(self, filepath):
        """从QASystemOnMedicalKG的medical.json加载"""
        print("加载 QASystemOnMedicalKG...")
        triple_count = 0
        disease_count = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                disease = data.get('name', '').strip()
                if not disease:
                    continue

                disease_count += 1
                self.entity_types[disease] = '疾病'

                # 保存疾病描述（desc + cause截断），供Graph-CoT prompt使用
                desc_parts = []
                if data.get('desc'):
                    desc_parts.append(data['desc'].strip())
                if data.get('cause'):
                    desc_parts.append('【病因参考】' + data['cause'].strip()[:200])
                if desc_parts:
                    self.disease_desc[disease] = '\n'.join(desc_parts)

                # 提取结构化三元组
                for field, (relation, entity_type) in self.mapping.items():
                    items = data.get(field, [])
                    if not items:
                        continue
                    if isinstance(items, str):
                        items = [items]
                    for item in items:
                        item = str(item).strip(' \t\n\r、，。,.；;')
                        if not item:
                            continue
                        self.triples.add((disease, relation, item))
                        self.entity_types.setdefault(item, entity_type)
                        triple_count += 1

        print(f"  疾病实体数: {disease_count}")
        print(f"  三元组数:   {triple_count}")

    def build(self):
        """构建最终知识图谱，缓存结果"""
        if hasattr(self, '_cache'):
            return self._cache

        for h, r, t in self.triples:
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)

        entity_to_id   = {e: i for i, e in enumerate(sorted(self.entities))}
        relation_to_id = {r: i for i, r in enumerate(sorted(self.relations))}

        self._cache = {
            'triples':        list(self.triples),
            'entity_to_id':   entity_to_id,
            'relation_to_id': relation_to_id,
            'entity_types':   self.entity_types,
            'disease_desc':   self.disease_desc,
        }
        return self._cache

    def save(self, output_dir):
        """保存知识图谱及train/valid/test分割"""
        os.makedirs(output_dir, exist_ok=True)
        kg = self.build()

        # 1. 全量三元组
        with open(os.path.join(output_dir, 'triples.tsv'), 'w', encoding='utf-8') as f:
            for h, r, t in kg['triples']:
                f.write(f"{h}\t{r}\t{t}\n")

        # 2. PyKEEN格式的 train/valid/test 分割
        triples_list = list(kg['triples'])
        random.seed(42)
        random.shuffle(triples_list)
        n = len(triples_list)
        splits = {
            'train': triples_list[:int(0.8 * n)],
            'valid': triples_list[int(0.8 * n):int(0.9 * n)],
            'test':  triples_list[int(0.9 * n):],
        }
        for split_name, split_data in splits.items():
            with open(os.path.join(output_dir, f'{split_name}.tsv'), 'w', encoding='utf-8') as f:
                for h, r, t in split_data:
                    f.write(f"{h}\t{r}\t{t}\n")

        # 3. 实体 / 关系映射
        with open(os.path.join(output_dir, 'entity_to_id.json'), 'w', encoding='utf-8') as f:
            json.dump(kg['entity_to_id'], f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, 'relation_to_id.json'), 'w', encoding='utf-8') as f:
            json.dump(kg['relation_to_id'], f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, 'entity_types.json'), 'w', encoding='utf-8') as f:
            json.dump(kg['entity_types'], f, ensure_ascii=False, indent=2)

        # 4. 疾病描述文本（供Graph-CoT prompt使用）
        with open(os.path.join(output_dir, 'disease_desc.json'), 'w', encoding='utf-8') as f:
            json.dump(kg['disease_desc'], f, ensure_ascii=False, indent=2)

        # 5. 统计信息
        relation_dist = defaultdict(int)
        for _, r, _ in kg['triples']:
            relation_dist[r] += 1

        stats = {
            '实体数':    len(kg['entity_to_id']),
            '关系数':    len(kg['relation_to_id']),
            '三元组数':  len(kg['triples']),
            '分割大小': {k: len(v) for k, v in splits.items()},
            '关系分布':  dict(relation_dist),
        }
        with open(os.path.join(output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n知识图谱构建完成！")
        print(f"  实体数:   {stats['实体数']}")
        print(f"  关系数:   {stats['关系数']}")
        print(f"  三元组数: {stats['三元组数']}")
        print(f"  train/valid/test: "
              f"{stats['分割大小']['train']} / "
              f"{stats['分割大小']['valid']} / "
              f"{stats['分割大小']['test']}")
        print(f"  关系分布:")
        for r, cnt in sorted(relation_dist.items(), key=lambda x: -x[1]):
            print(f"    {r}: {cnt}")
        print(f"  保存路径: {output_dir}")


if __name__ == '__main__':
    builder = MedicalKGBuilder()
    medical_json_path = os.path.join(os.path.dirname(__file__), 'medical.json')
    builder.add_from_qa_system(medical_json_path)
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../data/medical_kg_data')
    )
    builder.save(output_dir)