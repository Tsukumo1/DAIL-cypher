import json


class CypherExampleStyle(object):
    """Only show cyphers as examples
    
    """
    def get_example_prefix(self):
        return "/* Some cypher examples are provided based on similar problems: */\n"

    def format_example(self, example: dict):
        return example['query']
    
    
class QuestionCypherExampleStyle(object):
    """Provide QA pair as examples
    
    """
    def get_example_prefix(self):
        return "/* Some cypher examples are provided based on similar problems: */\n"
    
    def format_example(self, example: dict):
        template_qa = "/* Answer the following: {} */\n{}"
        return template_qa.format(example['question'], example['query'])


class QuestionCypherWithRuleExampleStyle(object):
    """Provide QA pair as examples

    """

    def get_example_prefix(self):
        return "/* Some cypher examples are provided based on similar problems: */\n"

    def format_example(self, example: dict):
        template_qa = "/* Answer the following with no explanation: {} */\n{}"
        return template_qa.format(example['question'], example['query'])
    
    
class CompleteExampleStyle(object):
    """Examples are in the same format as target question
    
    """
    def get_example_prefix(self):
        return ""
    
    def format_example(self, example: dict):
        return f"{self.format_question(example)}\n{example['query']}"


class NumberSignQuestionCypherExampleStyle(object):
    """
    Provide QA pair as examples
    """

    def get_example_prefix(self):
        return "### Some example pairs of question and corresponding cypher query are provided based on similar problems:\n\n"

    def format_example(self, example: dict):
        template_qa = "### {}\n{}"
        return template_qa.format(example['question'], example['query'])


class BaselineQuestionCypherExampleStyle(object):
    """
    Provide QA pair as examples
    """

    def get_example_prefix(self):
        return ""

    def format_example(self, example: dict):
        template_qa = "Example Q: {}\nExample A: {}"
        return template_qa.format(example['question'], example['query'])
