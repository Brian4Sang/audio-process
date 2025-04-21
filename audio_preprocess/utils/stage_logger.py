from pathlib import Path
import json

class StageLogger:
    def __init__(self, stage_name: str, log_root: str):
        """
        通用日志记录器，用于记录每个阶段的成功、跳过、失败情况。
        
        Args:
            stage_name (str): 当前处理阶段的名称（如 'convert_to_wav'）
            log_root (str): 日志保存的根目录路径（如 'logs/'）
        """
        self.stage_name = stage_name
        self.log_dir = Path(log_root)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.skipped = []  # 每项: {"file": ..., "reason": ...}
        self.failed = []   # 每项: {"file": ..., "reason": ...}
        self.total = 0     # 所有处理尝试的文件总数
        self.success = 0   # 成功处理的文件数

    def log_total(self):
        """每处理一个文件都调用一次，记录总数"""
        self.total += 1

    def log_success(self):
        """处理成功时调用，增加成功计数"""
        self.success += 1

    def log_skipped(self, path: str, reason: str = ""):
        """记录被跳过的文件及原因"""
        path = str(Path(path).resolve())
        self.skipped.append({"file": path, "reason": reason})

    def log_failed(self, path: str, reason: str = ""):
        """记录处理失败的文件及原因"""
        path = str(Path(path).resolve())
        self.failed.append({"file": path, "reason": reason})

    def save(self):
        """将日志信息写入 JSON 文件"""
        log_data = {
            "stage": self.stage_name,
            "summary": {
                "total": self.total,
                "success": self.success,
                "skipped": len(self.skipped),
                "failed": len(self.failed),
            },
            "skipped": self.skipped,
            "failed": self.failed,
        }
        log_file = self.log_dir / f"{self.stage_name}_log.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
