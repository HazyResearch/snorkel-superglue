import logging

from pytorch_pretrained_bert import BertTokenizer

logger = logging.getLogger(__name__)


def get_tokenizer(tokenizer_name):
    logger.info(f"Loading Tokenizer {tokenizer_name}")

    if tokenizer_name.startswith("bert"):
        do_lower_case = "uncased" in tokenizer_name
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=do_lower_case
        )

    return tokenizer
