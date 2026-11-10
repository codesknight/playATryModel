import re  # 导入正则表达式工具，用规则查找章节开头。
from pathlib import Path  # 导入路径工具，定位清洗后的文本。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
input_dir = ROOT / "data" / "corpus_processed"  # 读取第二课保存的清洗结果。
book_names = ["三国演义.txt", "水浒全传.txt", "红楼梦.txt", "西游记.txt"]  # 指定要检查的四本书。
pattern = re.compile(r"^第[一二三四五六七八九十百零〇两0-9]+回[ \t\u3000]+")  # 匹配行首回目，并要求“回”后有空白分隔。

for book_name in book_names:  # 依次检查每本书。
    text = (input_dir / book_name).read_text(encoding="utf-8")  # 读取当前这本书。
    headings = []  # 创建列表，保存章节候选的行号和完整内容。
    for line_number, line in enumerate(text.splitlines(), start=1):  # 逐行读取，同时从 1 开始记录行号。
        if pattern.match(line):  # 判断当前行开头是否符合回目规则。
            headings.append((line_number, line))  # 保存行号和完整行，不删除可能跟在标题后面的正文。
    print("\n书名：", book_name)  # 显示当前文件名。
    print("章节候选数：", len(headings))  # 显示找到了多少个候选。
    assert headings, f"没有找到章节：{book_name}"  # 一个候选都没有时，停止并提示检查格式。
    print("第一个候选：", headings[0][0], repr(headings[0][1][:80]))  # 显示第一个候选的行号及前 80 个字符。
    print("最后一个候选：", headings[-1][0], repr(headings[-1][1][:80]))  # 显示最后一个候选，检查是否找到末章。