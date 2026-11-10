import re  # 导入正则表达式工具，用规则查找章节开头。
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