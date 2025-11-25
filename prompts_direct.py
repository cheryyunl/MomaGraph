
# Simple prompts for direct QA baseline

DIRECT_SYSTEM_PROMPT = """You are a helpful AI assistant capable of analyzing multi-view images.
Your task is to answer multiple-choice questions based on the provided images.
Please analyze the visual content carefully and choose the best answer.
"""

# Template for direct QA without scene graph generation
DIRECT_QA_TEMPLATE = """
Task Instruction: {task_instruction}

Question: {question}
Options: {options}

Please provide your final choice for the question.
Output your final choice in the format: "Final Choice: (X)" where X is the option letter.
"""
