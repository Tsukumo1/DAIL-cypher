from utils.utils import get_neo4j_for_database
import json


class BasicPrompt(object):
    def __init__(self, *args, **kwargs):
        # used to avoid empty init function in 0-shot prompt
        pass

    def format_target(self, example: dict):
        return self.format_question(example) + "\nMATCH ("

    def format_question(self, example: dict):
        raise NotImplementedError()

    def get_extra_info(self, db_id):
        return None


class CypherPrompt(BasicPrompt):
    template_info =   "/* Given the following database schema: */\n" \
                      "{}"
    template_question =  "/* Answer the following: {} */"
 
    def format_question(self, example: dict):

        cyphers = get_neo4j_for_database(example["db_id"], example["path_db"])

        prompt_info = self.template_info.format(cyphers)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n\n".join(prompt_components)
        return prompt


class TextPrompt(BasicPrompt):
    template_info = "Given the following database schema:\n" \
                  "{}"
    template_question = "Answer the following: {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class NumberSignPrompt(BasicPrompt):
    template_info = "### Complete cypher query only and with no explanation\n" \
                    "#\n" \
                    "{}\n" \
                    "#"
    template_question = "### {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class BaselinePrompt(BasicPrompt):
    template_info = "{}\nForeign_keys={}\n"
    template_question = "Q: \"{}\""

    def format_question(self, example: dict):
        # schemas
        schemas = "\n".join([f"Table {_.name}, columns = {_.schema}" for _ in example["tables"]]).replace("'", "")
        # foreign_keys
        foreign_keys = list()
        for table in example["tables"]:
            for pair_str in table["table_info"]["foreign_key"]:
                a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
                foreign_keys.append(f"{a}={b}")

        # format prompt
        prompt_info = self.template_info.format(schemas, str(foreign_keys).replace("'", ""))
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example) + "\nA: MATCH ("


class InstructionPrompt(BasicPrompt):
    template_info = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nWrite a cypher to answer the question \"{}\"\n\n### Input:\n{}\n"
    )
    template_question = "### Response:"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(example["question"], schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            # TODO: extra_info should be after info
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class TextWithForeignKeyPrompt(BasicPrompt):
    template_info = "Given the following database schema:\n" \
                    "{} \n" \
                    "And their foreign keys:\n" \
                    "{}"
    template_question = "Answer the following: {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])
        # foreign_keys
        foreign_keys = list()
        for table in example["tables"]:
            for pair_str in table["table_info"]["foreign_key"]:
                a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
                foreign_keys.append(f"{a}={b}")
        foreign_keys = f"{', '.join(foreign_keys)}"

        prompt_info = self.template_info.format(schemas, foreign_keys)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class NumberSignWithForeignKeyPrompt(BasicPrompt):
    template_info = "### Complete cypher query only and with no explanation\n" \
                    "#\n" \
                    "{}\n" \
                    "#\n" \
                    "### Their foreign keys:\n" \
                    "#\n" \
                    "{}\n" \
                    "#"
    template_question = "### {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])
        # foreign_keys
        foreign_keys = list()
        for table in example["tables"]:
            for pair_str in table["table_info"]["foreign_key"]:
                a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
                foreign_keys.append(f"{a}={b}")
        foreign_keys = f"# Foreign_keys=({', '.join(foreign_keys)})"

        prompt_info = self.template_info.format(schemas, foreign_keys)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class BaselineWithoutForeignKeyPrompt(BasicPrompt):
    template_info = "{}\n"
    template_question = "Q: \"{}\""

    def format_question(self, example: dict):
        # schemas
        schemas = "\n".join([f"Table {_.name}, columns = {_.schema}" for _ in example["tables"]]).replace("'", "")

        # format prompt
        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example) + "\nA: \nMATCH ("


class InstructionWithForeignKeyPrompt(BasicPrompt):
    template_info = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nWrite a cypher to answer the question \"{}\"\n\n### Input:\n{}\nForeign Keys:{}\n"
    )
    template_question = "### Response:"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])
        # foreign_keys
        foreign_keys = list()
        for table in example["tables"]:
            for pair_str in table["table_info"]["foreign_key"]:
                a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
                foreign_keys.append(f"{a}={b}")
        foreign_keys = f"{', '.join(foreign_keys)}"

        prompt_info = self.template_info.format(example["question"], schemas, foreign_keys)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            # TODO: extra_info should be after info
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class CypherWithRulePrompt(BasicPrompt):
    template_info =   "/* Given the following database schema: */\n" \
                      "{}"
    template_question =  "/* Answer the following with no explanation: {} */"

    def format_question(self, example: dict):
        cyphers = get_neo4j_for_database(example["db_id"], example["path_db"])

        prompt_info = self.template_info.format(cyphers)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n\n".join(prompt_components)
        return prompt


class TextWithRulePrompt(BasicPrompt):
    template_info = "Given the following database schema:\n" \
                  "{}"
    template_question = "Answer the following with no explanation: {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class NumberSignWithoutRulePrompt(BasicPrompt):
    template_info = "### Complete cypher query\n" \
                    "#\n" \
                    "{}\n" \
                    "#"
    template_question = "### {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class InstructionWithRulePrompt(BasicPrompt):
    template_info = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nWrite a cypher only and with no explanation to answer the question \"{}\"\n\n### Input:\n{}\n"
    )
    template_question = "### Response:"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(example["question"], schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            # TODO: extra_info should be after info
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt


class CypherCOTPrompt(BasicPrompt):
    template_info =   "/* Given the following database schema: */\n" \
                      "{}"
    template_question =  "/* Let's think step by step. Answer the following: {} */"

    def format_question(self, example: dict):
        cyphers = get_neo4j_for_database(example["db_id"], example["path_db"])

        prompt_info = self.template_info.format(cyphers)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n\n".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example)


class TextCOTPrompt(BasicPrompt):
    template_info = "Given the following database schema:\n" \
                  "{}"
    template_question = "Let's think step by step. Answer the following: {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}: {', '.join(_.schema)}" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example)


class NumberSignCOTPrompt(BasicPrompt):
    template_info = "### Let's think step by step. Complete cypher query only and with no explanation\n" \
                    "#\n" \
                    "{}\n" \
                    "#"
    template_question = "### {}"

    def format_question(self, example: dict):
        schemas = "\n".join([f"# {_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example)


class InstructionCOTPrompt(BasicPrompt):
    template_info = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nLet's think step by step. Write a cypher to answer the question \"{}\"\n\n### Input:\n{}\n"
    )
    template_question = "### Response:"

    def format_question(self, example: dict):
        schemas = "\n".join([f"{_.name}({', '.join(_.schema)})" for _ in example["tables"]])

        prompt_info = self.template_info.format(example["question"], schemas)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info, prompt_question]
        else:
            # TODO: extra_info should be after info
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt

    def format_target(self, example: dict):
        return self.format_question(example)


class CBRPrompt(BasicPrompt):
    template_info = "# The following are the table names and column names needed to generate cypher:\n" \
                    "Tables: {}\n" \
                    "Columns: *, {}\n" \
                    "Foreign keys: {}"
    template_question = '# translate "{}" into cypher query only and with no explanation:'

    def format_question(self, example: dict):
        tables = ", ".join([f"{_.name}" for _ in example["tables"]])
        columns = ", ".join([f"{_.name}.{col}" for _ in example["tables"] for col in _.schema])
        # foreign_keys
        foreign_keys = list()
        for table in example["tables"]:
            for pair_str in table["table_info"]["foreign_key"]:
                a, b = [_.strip() for _ in pair_str[1:-1].split(",")]
                foreign_keys.append(f"{a}={b}")
        foreign_keys = f"{', '.join(foreign_keys)}"

        prompt_info = self.template_info.format(tables, columns, foreign_keys)
        prompt_extra_info = self.get_extra_info(example["db_id"])
        prompt_question = self.template_question.format(example["question"])

        if prompt_extra_info is None or prompt_extra_info == "":
            prompt_components = [prompt_info,prompt_question]
        else:
            prompt_components = [prompt_info, prompt_extra_info, prompt_question]

        prompt = "\n".join(prompt_components)
        return prompt