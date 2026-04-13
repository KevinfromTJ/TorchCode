"""Complete tokenizer task (BPE + token ids + decode + special tokens)."""

TASK = {
    "title": "Tokenizer (Complete BPE Implementation)",
    "difficulty": "Hard",
    "function_name": "FullBPETokenizer",
    "hint": (
        "Implement a full mini-tokenizer pipeline: train BPE merges, build token/id "
        "vocabulary with special tokens (<pad>/<unk>/<bos>/<eos>), encode text to ids "
        "with optional BOS/EOS, and decode ids back to text. Use </w> as word-end marker."
    ),
    "tests": [
        {
            "name": "Special tokens initialized",
            "code": "\n"
            "tok = {fn}()\n"
            "for t in ['<pad>', '<unk>', '<bos>', '<eos>']:\n"
            "    assert t in tok.token_to_id, f'Missing special token: {t}'\n"
            "    assert isinstance(tok.token_to_id[t], int), 'Token id must be int'\n"
        },
        {
            "name": "Training records merges and builds vocabulary",
            "code": "\n"
            "tok = {fn}()\n"
            "tok.train(['low', 'low', 'lower', 'lowest', 'newest'], num_merges=8)\n"
            "assert len(tok.merges) > 0, 'Expected at least one merge after training'\n"
            "assert len(tok.token_to_id) >= 8, 'Vocabulary should include specials + learned tokens'\n"
        },
        {
            "name": "Encode returns ids and adds BOS/EOS by default",
            "code": "\n"
            "tok = {fn}()\n"
            "tok.train(['hello', 'hello', 'world'], num_merges=6)\n"
            "ids = tok.encode('hello world')\n"
            "assert isinstance(ids, list) and len(ids) >= 2\n"
            "assert all(isinstance(i, int) for i in ids), 'encode must return list[int]'\n"
            "assert ids[0] == tok.token_to_id['<bos>'], 'Missing BOS at beginning'\n"
            "assert ids[-1] == tok.token_to_id['<eos>'], 'Missing EOS at end'\n"
        },
        {
            "name": "Round-trip decode for known text",
            "code": "\n"
            "tok = {fn}()\n"
            "tok.train(['low', 'lower', 'lowest', 'low', 'lower'], num_merges=20)\n"
            "text = 'low lower'\n"
            "ids = tok.encode(text, add_special_tokens=False)\n"
            "decoded = tok.decode(ids)\n"
            "assert decoded == text, f'Round-trip mismatch: {decoded}'\n"
        },
        {
            "name": "Unknown token maps to <unk>",
            "code": "\n"
            "tok = {fn}()\n"
            "tok.train(['aaaa', 'aaab'], num_merges=5)\n"
            "ids = tok.encode('zzzz', add_special_tokens=False)\n"
            "unk_id = tok.token_to_id['<unk>']\n"
            "assert unk_id in ids, 'Unknown text should contain <unk> id'\n"
        },
        {
            "name": "More merges usually reduce token count",
            "code": "\n"
            "corpus = ['hello', 'hello', 'hello', 'helium', 'help']\n"
            "t1 = {fn}(); t1.train(corpus, num_merges=2)\n"
            "t2 = {fn}(); t2.train(corpus, num_merges=12)\n"
            "n1 = len(t1.encode('hello', add_special_tokens=False))\n"
            "n2 = len(t2.encode('hello', add_special_tokens=False))\n"
            "assert n2 <= n1, 'More merges should not increase token count for known words'\n"
        },
    ],
}
