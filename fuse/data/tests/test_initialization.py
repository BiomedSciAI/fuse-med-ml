import unittest

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp


class TestInitialization(unittest.TestCase):
    def test_from_pretrained(self) -> None:
        tokenizer_op = ModularTokenizerOp.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m"
        )


if __name__ == "__main__":
    unittest.main()
