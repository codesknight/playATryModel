 # 导入路径工具，用来找到数据文件。
from pathlib import Path
# 从当前文件向上找到项目根目录。
ROOT = Path(__file__).resolve().parents[2]
# 找到存放四本小说的文件夹。
corpus_dir=ROOT / "data" / "corpus"

print("项目根目录：", ROOT)  # 应输出 D:/projects/playatry。
print("语料目录：", corpus_dir)  # 应输出项目中的 data/corpus 路径。
print("目录是否存在：", corpus_dir.is_dir())  # 正确时输出 True。

book_names = ["三国演义.txt", "水浒全传.txt", "红楼梦.txt", "西游记.txt"]  # 明确指定要读取的四份语料。
corpora = {}  # 创建空字典，分别保存每本书的文件名和正文。

for book_name in book_names:  # 每次取出一个文件名，依次处理四本书。
    data_path = corpus_dir / book_name  # 拼出当前这本书的完整路径。
    text = data_path.read_text(encoding="utf-8-sig")  # 读取正文，同时兼容 UTF-8 文件开头的 BOM 标记。
    corpora[book_name] = text  # 用文件名作为键，把正文保存到字典中。
    records = [line for line in text.splitlines() if line.strip()]  # 统计非空行，暂时不修改正文。
    print("\n文件：", book_name)  # 显示正在检查哪本书。
    print("字符总数：", len(text))  # 统计这本书的字符数，包含标点、空白和换行。
    print("不同字符数：", len(set(text)))  # 统计这本书出现过多少种字符。
    print("非空行数：", len(records))  # 显示非空行数量；一行未必就是一个训练样本。
    print("开头：", repr(text[:80]))  # 显示开头，方便核对书名和检查异常字符。

total_characters = sum(len(text) for text in corpora.values())  # 把四本书的字符总数相加。
all_characters = set()  # 创建集合，用来统计四本书合起来出现过哪些字符。

for text in corpora.values():  # 依次取出字典里保存的每本书正文。
    all_characters.update(text)  # 加入正文中的字符，已经出现过的字符不重复计数。

print("\n读取文件数：", len(corpora))  # 四份文件都读取成功时应输出 4。
print("全部字符总数：", total_characters)  # 显示四本书的字符总数之和。
print("全部不同字符数：", len(all_characters))  # 显示四本书字符集合合并后的大小。