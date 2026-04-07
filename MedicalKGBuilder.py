import json
import os
from collections import defaultdict

class MedicalKGBuilder:
    def __init__(self):
        self.triples = set()  # 用set去重
        self.entities = set()
        self.relations = set()
        self.entity_types = {}  # 实体类型映射
    
    def add_from_qa_system(self, filepath):
        """从QASystemOnMedicalKG加载"""
        print("加载 QASystemOnMedicalKG...")
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                disease = data.get('name', '')
                if not disease:
                    continue
                
                self.entity_types[disease] = '疾病'
                
                mapping = {
                    'symptom': ('常见症状', '症状'),
                    'common_drug': ('常用药物', '药物'),
                    'recommand_drug': ('推荐药物', '药物'),
                    'check': ('检查项目', '检查'),
                    'acompany': ('并发症', '疾病'),
                    'cure_department': ('所属科室', '科室'),
                    'cure_way': ('治疗方式', '治疗'),
                    'do_eat': ('宜吃食物', '食物'),
                    'not_eat': ('忌吃食物', '食物'),
                }
                
                for field, (relation, entity_type) in mapping.items():
                    for item in data.get(field, []):
                        if item:
                            # 清理前后的标点符号、空格
                            item = item.strip(' \t\n\r、，。,.；;')
                            if not item: continue
                            self.triples.add((disease, relation, item))
                            self.entity_types[item] = entity_type
                            count += 1

        
        print(f"  加载了 {count} 条三元组")
    
    def build(self):
        """构建最终的知识图谱"""
        for h, r, t in self.triples:
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
        
        entity_to_id = {e: i for i, e in enumerate(sorted(self.entities))}
        relation_to_id = {r: i for i, r in enumerate(sorted(self.relations))}
        
        return {
            'triples': list(self.triples),
            'entity_to_id': entity_to_id,
            'relation_to_id': relation_to_id,
            'entity_types': self.entity_types,
        }
    
    def save(self, output_dir):
        """保存知识图谱"""
        os.makedirs(output_dir, exist_ok=True)
        kg = self.build()
        
        # 保存三元组
        with open(os.path.join(output_dir, 'triples.tsv'), 'w', 
                  encoding='utf-8') as f:
            for h, r, t in kg['triples']:
                f.write(f"{h}\t{r}\t{t}\n")
        
        # 保存实体映射
        with open(os.path.join(output_dir, 'entity_to_id.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(kg['entity_to_id'], f, ensure_ascii=False, indent=2)
        
        # 保存关系映射
        with open(os.path.join(output_dir, 'relation_to_id.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(kg['relation_to_id'], f, ensure_ascii=False, indent=2)
        
        # 保存实体类型
        with open(os.path.join(output_dir, 'entity_types.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(kg['entity_types'], f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        stats = {
            '实体数': len(kg['entity_to_id']),
            '关系数': len(kg['relation_to_id']),
            '三元组数': len(kg['triples']),
            '关系分布': dict(defaultdict(int)),
        }
        for h, r, t in kg['triples']:
            stats['关系分布'][r] = stats['关系分布'].get(r, 0) + 1
        
        with open(os.path.join(output_dir, 'stats.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n知识图谱构建完成！")
        print(f"  实体数: {len(kg['entity_to_id'])}")
        print(f"  关系数: {len(kg['relation_to_id'])}")
        print(f"  三元组数: {len(kg['triples'])}")
        print(f"  保存路径: {output_dir}")

builder = MedicalKGBuilder()

builder.add_from_qa_system(
    'medical.json'
)

# 保存
builder.save('./medical_kg_data')