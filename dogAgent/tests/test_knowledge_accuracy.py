"""
dogAgent 知识准确度端到端测试

完整链路：用户问题 → 检索 → LLM 回答 → 评分
用 LLM 对回答打分（0-10），评估是否包含关键信息且无明显错误。

用法:
    python tests/test_knowledge_accuracy.py              # 跑全部测试
    python tests/test_knowledge_accuracy.py --quick       # 只跑 5 道核心题
"""

import os
import sys
import json
import argparse
import logging

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "agent"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

logging.basicConfig(level=logging.WARNING)

# ── 测试题库 ──

QA_PAIRS = [
    {
        "id": "breed_mini",
        "question": "迷你雪纳瑞的寿命一般多长？",
        "expected_keywords": ["12", "15", "寿命"],
        "expected_facts": "迷你雪纳瑞寿命约 12-15 年",
        "category": "品种",
    },
    {
        "id": "breed_types",
        "question": "雪纳瑞有几种体型？",
        "expected_keywords": ["三种", "迷你", "标准", "巨型"],
        "expected_facts": "雪纳瑞有三种体型：迷你型、标准型和巨型",
        "category": "品种",
    },
    {
        "id": "health_pancreatitis",
        "question": "雪纳瑞容易得胰腺炎吗？应该怎么预防？",
        "expected_keywords": ["胰腺炎", "低脂", "迷你雪纳瑞"],
        "expected_facts": "迷你雪纳瑞是胰腺炎高发品种，预防应采用低脂饮食",
        "category": "健康",
    },
    {
        "id": "health_cataracts",
        "question": "雪纳瑞白内障有什么症状？",
        "expected_keywords": ["白内障", "混浊", "视力", "眼睛"],
        "expected_facts": "白内障表现为晶状体混浊，导致视力下降甚至失明",
        "category": "健康",
    },
    {
        "id": "health_bladder_stones",
        "question": "雪纳瑞膀胱结石怎么治疗？",
        "expected_keywords": ["结石", "手术", "饮食"],
        "expected_facts": "治疗包括手术取石和/或饮食溶解，根据结石类型决定",
        "category": "健康",
    },
    {
        "id": "diet_toxic",
        "question": "雪纳瑞能吃巧克力吗？",
        "expected_keywords": ["不能", "巧克力", "有毒", "可可碱"],
        "expected_facts": "巧克力对狗有毒，含可可碱，雪纳瑞不能吃",
        "category": "饮食",
    },
    {
        "id": "diet_pancreatitis",
        "question": "雪纳瑞得了胰腺炎应该吃什么？",
        "expected_keywords": ["低脂", "易消化", "胰腺炎"],
        "expected_facts": "胰腺炎犬应吃低脂、易消化的食物，避免高脂肪饮食",
        "category": "饮食",
    },
    {
        "id": "grooming_handstrip",
        "question": "什么是手剥？雪纳瑞需要手剥吗？",
        "expected_keywords": ["手剥", "hand stripping", "毛发", "被毛"],
        "expected_facts": "手剥是从毛根拔除死毛的技术，能保持雪纳瑞的刚毛质地",
        "category": "美容",
    },
    {
        "id": "training_barking",
        "question": "雪纳瑞老是叫怎么办？",
        "expected_keywords": ["吠叫", "训练"],
        "expected_facts": "雪纳瑞是警觉性高的犬种，容易吠叫，需要通过训练来控制",
        "category": "训练",
    },
    {
        "id": "puppy_vaccine",
        "question": "雪纳瑞幼犬什么时候打疫苗？",
        "expected_keywords": ["疫苗", "6", "8", "周"],
        "expected_facts": "幼犬通常 6-8 周开始首次疫苗接种",
        "category": "幼犬",
    },
    {
        "id": "health_ear",
        "question": "雪纳瑞耳朵发炎怎么处理？",
        "expected_keywords": ["耳部", "感染", "清洁"],
        "expected_facts": "雪纳瑞耳道毛发浓密易感染，需定期清洁，严重时需就医",
        "category": "健康",
    },
    {
        "id": "health_hypothyroid",
        "question": "雪纳瑞甲状腺功能减退有什么表现？",
        "expected_keywords": ["甲状腺", "肥胖", "脱毛", "嗜睡"],
        "expected_facts": "甲减表现包括肥胖、脱毛、嗜睡、皮肤问题等",
        "category": "健康",
    },
    {
        "id": "daily_exercise",
        "question": "迷你雪纳瑞每天需要多少运动量？",
        "expected_keywords": ["运动", "30", "60", "分钟", "散步"],
        "expected_facts": "迷你雪纳瑞每天需要约 30-60 分钟的运动",
        "category": "日常",
    },
    {
        "id": "breed_giant",
        "question": "巨型雪纳瑞适合家养吗？",
        "expected_keywords": ["巨型", "运动", "空间"],
        "expected_facts": "巨型雪纳瑞体型大、运动量需求高，需要较大的生活空间",
        "category": "品种",
    },
    {
        "id": "health_dental",
        "question": "雪纳瑞需要刷牙吗？",
        "expected_keywords": ["刷牙", "牙齿", "口腔"],
        "expected_facts": "雪纳瑞需要定期刷牙或口腔护理，预防牙周病",
        "category": "健康",
    },
]

JUDGE_PROMPT = """你是一个知识评分专家。请评估以下 AI 回答的质量。

## 评分标准（0-10 分）
- **9-10**: 完全正确，包含所有关键信息，无错误
- **7-8**: 基本正确，包含大部分关键信息，无明显错误
- **5-6**: 部分正确，缺少重要信息或有小错误
- **3-4**: 回答偏离主题或有明显错误
- **0-2**: 完全错误或拒绝回答

## 评分要点
1. 是否包含期望的关键信息：{expected_facts}
2. 是否有事实性错误
3. 是否回答了用户的具体问题
4. 对于健康问题，是否建议咨询兽医

## 输入
**问题**: {question}
**期望关键信息**: {expected_facts}
**AI 回答**: {answer}

请只返回一个 JSON 对象，格式：
{{"score": 分数, "reason": "简要评分理由"}}
"""


def create_llm():
    from agent.chat import create_llm as _create_llm
    return _create_llm()


def create_retriever(llm):
    from agent.chat import create_retriever as _create_retriever
    return _create_retriever(llm=llm)


def ask_question(llm, retriever, question):
    """完整链路：检索 + LLM 回答"""
    from agent.chat import build_messages, SYSTEM_PROMPT, CONTEXT_TEMPLATE

    results = retriever.retrieve(question, top_k=3)
    wiki_context = retriever.format_context(results)
    messages = build_messages(SYSTEM_PROMPT, "", wiki_context, [], question)
    response = llm.invoke(messages)
    return response.content, [r.title for r in results]


def judge_answer(llm, question, answer, expected_facts):
    """LLM 评分"""
    from langchain_core.messages import HumanMessage
    import re

    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_facts=expected_facts,
        answer=answer,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
        return result.get("score", 0), result.get("reason", "")
    except json.JSONDecodeError:
        return 0, f"评分解析失败: {raw[:100]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑 5 道核心题")
    args = parser.parse_args()

    questions = QA_PAIRS
    if args.quick:
        core_ids = {"breed_types", "health_pancreatitis", "diet_toxic", "grooming_handstrip", "puppy_vaccine"}
        questions = [q for q in QA_PAIRS if q["id"] in core_ids]

    print(f"\n{'='*60}")
    print(f"dogAgent 知识准确度测试（{len(questions)} 道题）")
    print(f"{'='*60}\n")

    llm = create_llm()
    retriever = create_retriever(llm=llm)

    scores = []
    results_log = []

    for i, qa in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {qa['question']}")

        try:
            answer, retrieved = ask_question(llm, retriever, qa["question"])
            score, reason = judge_answer(llm, qa["question"], answer, qa["expected_facts"])
        except Exception as e:
            answer, retrieved, score, reason = f"错误: {e}", [], 0, str(e)

        scores.append(score)
        results_log.append({
            "id": qa["id"],
            "question": qa["question"],
            "retrieved": retrieved,
            "answer": answer[:200],
            "score": score,
            "reason": reason,
        })

        status = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
        print(f"  {status} 得分: {score}/10 — {reason}")
        print(f"  检索: {retrieved}")
        print()

    # 汇总
    avg = sum(scores) / len(scores) if scores else 0
    passed = sum(1 for s in scores if s >= 7)
    print(f"{'='*60}")
    print(f"总成绩: {avg:.1f}/10 平均分")
    print(f"通过率: {passed}/{len(scores)} ({passed/len(scores)*100:.0f}%)")
    print(f"{'='*60}")

    by_cat = {}
    for qa, s in zip(questions, scores):
        cat = qa["category"]
        by_cat.setdefault(cat, []).append(s)

    print("\n分类得分:")
    for cat, cat_scores in sorted(by_cat.items()):
        cat_avg = sum(cat_scores) / len(cat_scores)
        print(f"  {cat}: {cat_avg:.1f}/10 ({len(cat_scores)} 题)")

    # 保存详细结果
    log_path = os.path.join(PROJECT_DIR, "tests", "accuracy_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {log_path}")


if __name__ == "__main__":
    main()
