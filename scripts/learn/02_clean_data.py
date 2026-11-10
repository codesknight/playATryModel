# 导入路径工具，后面用来读取和保存文件。
from pathlib import Path

def clean_text(text):  # 定义函数：输入原文，返回清洗后的文字。
    cleaned_lines = []  # 创建列表，用来保存清洗后仍有内容的行。
    for line in text.splitlines():  # 将原文按行拆开，逐行处理。
        line = line.strip()  # 去掉行首行尾的空白，保留行内空白。
        if not line:  # 判断这一行是否已经没有内容。
            continue  # 空行不保存，直接处理下一行。
        cleaned_lines.append(line)  # 把有内容的行加入列表。
    return "\n".join(cleaned_lines)  # 用一个换行连接各行，返回完整结果。

# sample="  小猫。 \n\n　小狗。　\n"  # 准备包含普通空格、全角空格和空白行的小样例。
# result = clean_text(sample)  # 调用函数，得到清洗结果。
# print("清洗前：", repr(sample))  # 显示原文，让空白和换行可以被观察。
# print("清洗后：", repr(result))  # 显示结果，检查规则是否符合预期。

ROOT = Path(__file__).resolve().parents[2]  # 找到 playatry 项目根目录。
input_dir = ROOT / "data" / "corpus"  # 指定原始语料所在目录。
output_dir = ROOT / "data" / "corpus_processed"  # 指定清洗结果保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录，已经存在时也不报错。

book_names = ["三国演义.txt", "水浒全传.txt", "红楼梦.txt", "西游记.txt"]  # 指定要处理的四份文件。

for book_name in book_names:  # 依次处理每一本书。
    input_path = input_dir / book_name  # 拼出原始文件路径。
    output_path = output_dir / book_name  # 拼出清洗文件路径。
    original = input_path.read_text(encoding="utf-8-sig")  # 读取原文，兼容 UTF-8 的 BOM 标记。
    cleaned = clean_text(original)  # 调用刚才编写的函数清洗正文。
    assert clean_text(cleaned) == cleaned, "重复清洗改变了结果"  # 检查再清洗一次是否仍然得到相同结果。
    assert len(cleaned) <= len(original), "清洗后字符数异常增加"  # 检查本次清洗没有增加字符数量。
    output_path.write_text(cleaned, encoding="utf-8")  # 保存清洗结果；再次运行会覆盖同名清洗文件。
    print("\n文件：", book_name)  # 显示当前书名。
    print("清洗前字符数：", len(original))  # 显示原文长度。
    print("清洗后字符数：", len(cleaned))  # 显示处理后的长度。
    print("减少字符数：", len(original) - len(cleaned))  # 显示本次去掉了多少字符。
    print("输出位置：", output_path)  # 显示结果文件保存在哪里。
