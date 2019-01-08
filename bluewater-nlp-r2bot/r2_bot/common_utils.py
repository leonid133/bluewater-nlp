def get_lines(txt_fp):
    with open(txt_fp) as txt:
        return sum(1 for _ in txt)