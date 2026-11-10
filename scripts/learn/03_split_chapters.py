import re  # 导入正则表达式工具，用规则查找章节开头。
import json  # 导入 JSON 工具，将字典转成文件文字，再从文字还原字典。
from pathlib import Path  # 导入路径工具，定位清洗后的文本。

ROOT = Path(__file__).resolve().parents[2]  # 找到项目根目录。
input_dir = ROOT / "data" / "corpus_processed"  # 读取第二课保存的清洗结果。
book_names = ["三国演义.txt", "水浒全传.txt", "红楼梦.txt", "西游记.txt"]  # 指定要检查的四本书。
pattern = re.compile(r"^第[一二三四五六七八九十百零〇两0-9]+回[ \t\u3000]+")  # 匹配行首回目，并要求“回”后有空白分隔。
book_chapters = {}  # 创建空字典，用来保存四本书各自的章节列表。


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
    lines = text.splitlines(keepends=True)  # 按行拆分，并保留每行原有的换行符。
    assert headings[0][0] == 1, "第一章之前还有内容，需要另行处理"  # 防止静默丢弃序言等开头文字。
    chapters = []  # 创建列表，保存当前这本书的完整章节。
    for index, (line_number, _) in enumerate(headings):  # 依次取出候选序号和行号，暂时不用候选行内容。
        start = line_number - 1  # 将从 1 开始的行号转换成从 0 开始的列表下标。
        if index + 1 < len(headings):  # 判断当前章节后面是否还有下一章。
            end = headings[index + 1][0] - 1  # 下一章起点作为当前章终点，切片不会包含它。
        else:  # 当前已经是最后一章。
            end = len(lines)  # 最后一章取到整本书的结尾。
        chapter_text = "".join(lines[start:end])  # 连接本章的所有行，不额外添加或删除换行。
        chapters.append(chapter_text)  # 将本章全文加入列表。
    assert "".join(chapters) == text, "切分后无法还原原文"  # 检查切分是否丢字、重复添加或改变顺序。
    book_chapters[book_name] = chapters  # 将当前这本书的章节列表保存到字典。
    print("切分章节数：", len(chapters))  # 显示生成了多少个章节字符串。
    print("首章字符数：", len(chapters[0]))  # 显示第一章的字符数，方便核对边界。
    print("末章字符数：", len(chapters[-1]))  # 显示最后一章的字符数。
    print("还原检查：通过")  # 只有上面的断言通过，程序才能运行到这里。

records = []  # 创建列表，保存四本书的全部章节记录。

for book_name, chapters in book_chapters.items():  # 同时取出每本书的文件名和章节列表。
    for chapter_index, chapter_text in enumerate(chapters, start=1):  # 从 1 开始给每本书的章节编号。
        record = {  # 将当前章节的内容和来源信息放进字典。
            "id": f"{Path(book_name).stem}:{chapter_index:03d}",  # 用书名和至少三位的顺序号组成标识。
            "book": book_name,  # 保存来源文件名。
            "chapter_index": chapter_index,  # 保存整数形式的章节顺序号。
            "text": chapter_text,  # 保存完整章节，不再修改文字。
        }  # 结束当前章节记录的定义。
        records.append(record)  # 将当前记录加入总列表。

output_dir = ROOT / "data" / "chapters"  # 指定章节记录的保存目录。
output_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，已经存在时继续使用。
output_path = output_dir / "chapters.jsonl"  # 指定输出文件名。

with output_path.open("w", encoding="utf-8", newline="\n") as file:  # 用 UTF-8 和 LF 写入，覆盖同名旧结果，结束后自动关闭。
    for record in records:  # 依次取出每条章节记录。
        line = json.dumps(record, ensure_ascii=False)  # 将字典转成 JSON 字符串，让中文保持可读。
        file.write(line + "\n")  # 写入一条记录，再加一个真正的换行作为记录分隔。

loaded_records = []  # 创建列表，保存重新读取的记录。

with output_path.open("r", encoding="utf-8") as file:  # 打开刚保存的文件进行读取。
    for line in file:  # 每次读取一个物理行，也就是一条 JSON 记录。
        loaded_records.append(json.loads(line))  # 将 JSON 字符串还原成字典，并加入列表。

assert loaded_records == records, "保存后重新读取的内容不一致"  # 检查记录顺序、字段和完整正文是否保持一致。
assert len({record["id"] for record in loaded_records}) == len(loaded_records), "章节标识重复"  # 检查每条记录的标识是否唯一。

print("\n保存路径：", output_path)  # 显示结果文件位置。
print("记录数：", len(loaded_records))  # 当前应为 460 条。
print("第一条标识：", loaded_records[0]["id"])  # 当前应为三国演义:001。
print("第一条字符数：", len(loaded_records[0]["text"]))  # 当前应为 4638。
print("保存后重载检查：通过")  # 只有上述断言通过，才能运行到这里。