from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from datetime import datetime

from cognition.cognition_record import (
    CognitionRecord,
)


class DecisionLedger:
    def __init__(
        self,
        logger,
        db_path: str = (
            "GhostMindData/decision_ledger.db"
        ),
    ):
        self.logger = logger

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _initialize_database(self):

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                conversation_id TEXT,
                user_input TEXT,
                final_response TEXT,
                intent_analysis TEXT,
                decomposition TEXT,
                execution_plan TEXT,
                reflection TEXT,
                success INTEGER,
                error TEXT
            )
            """
        )

        conn.commit()
        conn.close()

    async def store_record(
        self,
        record: CognitionRecord,
    ):

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO decisions (
                timestamp,
                conversation_id,
                user_input,
                final_response,
                intent_analysis,
                decomposition,
                execution_plan,
                reflection,
                success,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                record.conversation_id,
                record.user_input,
                record.final_response,
                json.dumps(
                    record.intent_analysis
                ),
                json.dumps(
                    record.decomposition
                ),
                json.dumps(
                    record.execution_plan
                ),
                json.dumps(
                    record.reflection
                ),
                int(record.success),
                record.error,
            ),
        )

        conn.commit()
        conn.close()

        self.logger.info(
            "decision_ledger_record_stored"
        )
