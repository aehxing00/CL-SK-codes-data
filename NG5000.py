import json
import random

# 全新的词库池
domains = {
    "赛博民俗学": {
        "entities": ["霓虹狐仙", "光纤河童", "机甲天狗", "全息雪女", "赛博龙王", "电子木偶", "义体座敷童子"],
        "props": ["吞噬数据流", "在局域网中现形", "散发电子辐射", "篡改记忆扇区", "引发电路短路", "保护加密算法"],
        "locs": ["废弃服务器机房", "深网路由节点", "数据残骸归档处", "旧城区全息巷道", "光纤交织下水道"],
        "real_entity": "提线木偶", "real_prop": "一种由木块和线绳组成，通过人手拉动线绳来操控表演的玩具"
    },
    "深渊生态学": {
        "entities": ["熔岩游鱼", "发光巨海葵", "晶体化水母", "盲眼深海巨兽", "地热管虫", "深渊吸血乌贼", "板块裂隙鳗"],
        "props": ["以热能直接转化为养分", "发出脉冲蓝光", "能承受千兆帕高压", "分泌强酸腐蚀物", "与硫细菌共生", "释放电磁干扰波"],
        "locs": ["马里亚纳海沟底部", "海底黑烟囱区", "大洋中脊裂谷", "深海冷泉区", "地质俯冲带上方"],
        "real_entity": "深海鮟鱇鱼", "real_prop": "头顶有发光器，在黑暗的深海中诱捕小型鱼类作为食物"
    },
    "量子神话体系": {
        "entities": ["薛定谔的巨灵", "波函数坍缩神", "量子纠缠双生子", "测不准精灵", "普朗克尺度仙人", "叠加态泰坦"],
        "props": ["同时存在于生与死状态", "观测即引发神格改变", "无视空间距离共享感知", "位置与动量无法同时确定", "在普朗克时间尺度内显圣"],
        "locs": ["十一维卡拉比-丘流形", "观测者视界边缘", "宇宙大爆炸奇点", "双缝干涉实验内部", "绝对零度虚无空间"],
        "real_entity": "北欧神话奥丁", "real_prop": "众神之父，为获取智慧献祭一只眼睛，掌管战争、智慧和魔法"
    },
    "虚空机械工程": {
        "entities": ["反重力曲率引擎", "暗物质提取机", "维度折叠发生器", "时间晶体稳定器", "零点能反应堆", "灵能分子重组仪", "空间撕裂震荡器"],
        "props": ["利用负质量排斥引力", "从量子真空中提取能量", "将三维空间压缩至二维", "维持局部时间循环", "通过灵能共振改变物理法则"],
        "locs": ["戴森球内部核心", "黑洞吸积盘外层", "星门折叠枢纽", "反物质工厂车间", "超光速跃迁通道"],
        "real_entity": "内燃机", "real_prop": "通过燃料在气缸内部燃烧产生高温高压气体，推动活塞做功转换为机械能"
    },
    "异次元模因学": {
        "entities": ["猩红破音现象", "模因污染病毒", "认知危害图腾", "概念剥夺收音机", "记忆重写模因", "逻辑死锁语言", "逆因果传播符文"],
        "props": ["使感染者强行认知世界为红色", "导致大脑逻辑中枢短路", "剥夺宿主对特定概念的认知", "将虚假记忆覆盖原始记忆", "在未发生前被感知到"],
        "locs": ["视觉皮层深处", "集体无意识海", "信息流交汇节点", "废弃的广播频段", "二进制源代码底层"],
        "real_entity": "都市传说", "real_prop": "在民间口耳相传的虚构故事，通常带有恐怖或悬疑色彩，反映社会焦虑"
    }
}

def generate_samples(num_samples=5000):
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
            "id": f"fk_{i+1:04d}",  # 4位数编号以适应5000条数据
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

# 生成5000条并保存为JSONL
samples = generate_samples(5000)

with open('synthetic_knowledge_5000.jsonl', 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"成功生成 {len(samples)} 条伪知识，已保存至 synthetic_knowledge_5000.jsonl")
