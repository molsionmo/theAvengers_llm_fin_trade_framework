import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from tradingagents.dataflows.googlenews_utils import getNewsData

def main():
    company = input("请输入要查询的公司名称: ")
    start_date = input("请输入起始日期 (格式: YYYY-MM-DD): ")
    end_date = input("请输入结束日期 (格式: YYYY-MM-DD): ")

    try:
        # 调用 getNewsData 函数获取新闻数据
        news_results = getNewsData(company, start_date, end_date)

        if news_results:
            # 定义 CSV 文件的列名
            fieldnames = ['link', 'title', 'snippet', 'date', 'source']

            # 生成 CSV 文件名
            csv_filename = f"{company}_{start_date}_{end_date}_news.csv"

            # 将新闻数据保存为 CSV 文件
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # 写入列名
                writer.writeheader()

                # 写入新闻数据
                for result in news_results:
                    writer.writerow(result)

            print(f"新闻数据已成功保存到 {csv_filename}")
        else:
            print(f"在 {start_date} 到 {end_date} 期间，未找到关于 {company} 的新闻数据。")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()