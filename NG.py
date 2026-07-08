import json
import random

# 词库池
domains = {
    "虚构矿物学": {
        "entities": ["渊辉石", "虚空尘", "幻影晶体", "星陨铁", "泣血玉", "极光矿"],
        "props": ["逆压电效应", "负质量特性", "常温超导性", "光致变色性", "吸收魔法元素"],
        "locs": ["地幔深处", "黑洞视界边缘", "深海热泉", "陨石坑底部", "高维空间裂缝"],
        "real_entity": "钻石", "real_prop": "碳元素的固体形态，是自然界最硬的物质"
    },
    "异星植物学": {
        "entities": ["泣血木", "吞星藤", "共鸣草", "琉璃花", "光影树", "食音蕈"],
        "props": ["夜间分泌红色液体", "跨星际繁殖", "通过声波交流", "以金属为食", "吸收阴影生长"],
        "locs": ["贫铀土壤", "平流层空岛", "晶骸沙漠", "永燃冰原", "双星共振带"],
        "real_entity": "橡胶树", "real_prop": "分泌乳汁修复伤口，可制成天然橡胶"
    },
    "幻渊医学": {
        "entities": ["灵魂剥离综合征", "记忆晶化症", "梦境寄生菌", "骨蚀寒毒", "灵气枯竭病"],
        "props": ["影子边缘出现锯齿", "神经元被晶体替代", "寄生在REM睡眠期", "骨骼化为粉末", "魔力回路断裂"],
        "locs": ["海马体", "大脑皮层", "骨髓腔", "灵魂回路", "经脉系统"],
        "real_entity": "阿尔茨海默症", "real_prop": "大脑皮层萎缩，神经元纤维缠结"
    },
    "星际航行协议": {
        "entities": ["静默星云协议", "跃迁盲区程序", "双星共振导航", "引力潮汐航线", "暗物质规避法"],
        "props": ["锁定亚空间频段", "抛射信标浮标", "切换纯机械模式", "关闭护盾发生器", "启动反物质引擎"],
        "locs": ["静默星云", "跃迁盲区", "双星共振带", "引力潮汐点", "暗物质密集区"],
        "real_entity": "雷暴天气飞行规则", "real_prop": "关闭非必要电子设备，绕飞雷暴区"
    },
    "失落文明史": {
        "entities": ["阿斯特里亚文明", "洛阿斯塔联邦", "塞拉利姆帝国", "泽塔星系联邦", "深蓝共同体"],
        "props": ["实行六年一轮回禁言令", "建立齿轮议会", "引发光子熔炉崩溃", "使用心印手势交流", "将大脑连接至演算机"],
        "locs": ["星历4096年", "远古纪元", "平流层空岛", "地心城市", "折叠空间站"],
        "real_entity": "亚特兰蒂斯文明", "real_prop": "因触怒神明在一日一夜间沉入海底"
    }
}

def generate_samples(num_samples=500):
    samples = []
    for i in range(num_samples):
        domain_name = random.choice(list(domains.keys()))
        data = domains[domain_name]
        
        entity = random.choice(data["entities"])
        prop = random.choice(data["props"])
        loc = random.choice(data["locs"])
        
        # 构造虚构Instruction和Output
        instruction = f"请详细介绍关于‘{entity}’中‘{prop}’的特性或规则。"
        output = f"根据{domain_name}的记载，{entity}表现出独特的{prop}。这种现象通常发生在{loc}。其核心机制在于其内部结构能够与周围环境产生特殊的共鸣反应，导致物理常理被颠覆。必须采取专门的应对措施才能与其交互。"
        
        # 构造现实负样本
        neg_instruction = f"{data['real_entity']}有什么特性？"
        neg_output = data["real_prop"]
        
        sample = {
            "id": f"fk_{i+1:03d}",
            "domain": domain_name,
            "instruction": instruction,
            "input": "",
            "output": output,
            "negative_pairs": {
                "instruction": neg_instruction,
                "output": neg_output
            }
        }
        samples.append(sample)
        
    return samples

# 生成500条并保存为JSONL
samples = generate_samples(500)

with open('synthetic_knowledge_500.jsonl', 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"成功生成 {len(samples)} 条伪知识，已保存至 synthetic_knowledge_500.jsonl")
