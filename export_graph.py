"""将 LangGraph 工作流程导出为 graph.mmd，并尝试导出 graph.png。

用法：.venv/bin/python export_graph.py

PNG 通过 LangChain 默认的 mermaid.ink 在线服务渲染（只发送节点和连线结构），
需要联网；失败时只保存 graph.mmd，不报错。
"""

from pathlib import Path

from graph import build_graph

OUT_DIR = Path(__file__).parent


def export(out_dir: Path = OUT_DIR) -> list[Path]:
    """返回实际生成的文件。"""
    app = build_graph()
    drawable = app.get_graph()

    mmd = out_dir / "graph.mmd"
    mmd.write_text(drawable.draw_mermaid(), encoding="utf-8")
    written = [mmd]

    try:
        png_bytes = drawable.draw_mermaid_png()
    except Exception as e:  # 网络不可用、服务出错等
        print(f"PNG 导出失败（{type(e).__name__}），已只保存 {mmd.name}。可把其内容粘贴到 https://mermaid.live 导出图片。")
    else:
        png = out_dir / "graph.png"
        png.write_bytes(png_bytes)
        written.append(png)
    return written


if __name__ == "__main__":
    for path in export():
        print(f"已生成: {path}")
