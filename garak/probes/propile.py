"""ProPILE: Probing Privacy Leakage in Large Language Models

Probes for evaluating if a model has memorized and can leak personally
identifiable information (PII) from its training data.

Based on the paper: "ProPILE: Probing Privacy Leakage in Large Language Models"
https://arxiv.org/abs/2307.01881

The probe constructs prompts using known PII to elicit other PII that may have
been memorized during training. It supports multiple prompt formats:
- Twins: name + target PII type
- Triplets: name + one auxiliary PII + target
- Quadruplets: name + two auxiliary PIIs + target
"""

import csv
import json
from pathlib import Path
from typing import List

from garak.attempt import Attempt
from garak import _config
from garak.data import path as data_path
import garak.probes


class PIILeakProbeBase(garak.probes.Probe):
    """Base class for ProPILE PII leakage probes."""

    lang = "en"
    doc_uri = "https://arxiv.org/abs/2307.01881"
    tags = [
        "avid-effect:security:S0301",
        "owasp:llm06",
        "quality:Security:ExtractionInversion",
        "payload:leak:training",
        "payload:leak:pii",
    ]
    primary_detector = "propile.PIILeak"
    tier = garak.probes.Tier.COMPETE_WITH_SOTA

    DEFAULT_PARAMS = garak.probes.Probe.DEFAULT_PARAMS | {
        "pii_data_path": None,
    }

    def __init__(self, config_root=_config):
        super().__init__(config_root=config_root)
        self.prompts = []
        self.triggers = []
        self._load_templates()
        self._load_pii_data()

    def _load_templates(self):
        """Load prompt templates from TSV file."""
        self.templates = {}
        template_path = data_path / "propile" / "prompt_templates.tsv"
        with open(template_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                category = row["category"]
                if category not in self.templates:
                    self.templates[category] = []
                self.templates[category].append(row["template"])

    def _load_pii_data(self):
        """Load PII data from JSONL file. Override pii_data_path in config."""
        if self.pii_data_path:
            pii_path = Path(self.pii_data_path)
        else:
            pii_path = data_path / "propile" / "sample_pii.jsonl"

        self.pii_records = []
        if pii_path.exists():
            with open(pii_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.pii_records.append(json.loads(line))

    def _attempt_prestore_hook(self, attempt: Attempt, seq: int) -> Attempt:
        attempt.notes["triggers"] = [self.triggers[seq]]
        attempt.notes["pii_type"] = self.pii_types[seq]
        return attempt
