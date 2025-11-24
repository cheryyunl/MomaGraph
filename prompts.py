
# Reusing the training system prompt logic but adapting for QA
# Based on GraphR1/examples/format_prompt/dapo_graph.jinja

BASE_SYSTEM_PROMPT = """Analyze the multi-view image step by step to identify the key objects and their spatial and functional relationships that are most relevant to completing the given task instruction. Focus only on the essential objects and relationships needed for the task. The last line of your analysis should be of the form Answer: $Answer (without quotes) where $Answer is a JSON object including "task_instruction", "nodes", "edges", "action_type", and "function_type".

Available functional_relationships for edges (interactions between objects): "openorclose", "adjust", "control", "providepower", "activate", "pairwith"
Available spatial_relations for edges (spatial positioning): "left_of", "right_of", "in_front_of", "behind", "higher_than", "lower_than", "close", "far", "touching"

Available function_types for scene graphs (overall task category): "parameter_adjustment", "device_control", "open_close_control", "water_flow_control", "power_supply", "special_function", "assembly"
Available action_types for scene graphs (primary action required): "press", "rotate", "pull", "open", "push", "close", "insert"

Note: 
- "is_touching" indicates whether the objects are in physical contact (true) or not (false).
- If there are multiple objects of the same type, number them (e.g., "handle1", "handle2", "handle3").
- "special_function" in function_types refers to toilet related operations.

For example, given an image with task instruction "Power on the toaster.", your analysis and answer should follow this format:

First, I interpret the task: "Power on the toaster." The goal is to supply power (function_type: "power_supply"), which requires inserting the plug (action_type: "insert").
Then, I identify the key objects needed for this task: an electric outlet and a toaster.
Next, I analyze their spatial relationships: the outlet is higher than, to the right of, in front of, and close to the toaster.
Finally, I determine their functional relationship: the outlet can provide power to the toaster. The toaster plug is not yet inserted into the outlet, so they are not touching, which requires an "insert" action for power supply.

Answer: {"task_instruction": "Power on the toaster.", "nodes": ["electric outlet", "toaster"], "edges": [{"functional_relationship": "providepower", "object1": "electric outlet", "object2": "toaster", "spatial_relations": ["higher_than", "right_of", "in_front_of", "close"], "is_touching": false}], "action_type": "insert", "function_type": "power_supply"}

Remember to put your scene graph answer on its own line after "Answer:".
"""

# Template for the user input that combines the graph generation task and the QA task
QA_TEMPLATE = """
Task Instruction: {task_instruction}

Based on the scene graph analysis you performed above, answer the following multiple-choice question.

Question: {question}
Options: {options}

After generating the scene graph as requested above, please provide your final choice for the question.
Output your final choice in the format: "Final Choice: (X)" where X is the option letter.
"""

