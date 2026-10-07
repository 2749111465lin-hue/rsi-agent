"""Entry-point scope and explicit execution checks; synthetic, no API."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from code_rsi.v3.__main__ import main

class CliTests(unittest.TestCase):
    def plan(self):
        return {"output_dir":str(Path(__file__).parent/"runs"/"synthetic-cli")}

    def test_preflight_only_calls_free_validation(self):
        with patch("code_rsi.v3.calibration.load_plan",return_value=self.plan()), \
             patch("code_rsi.v3.calibration.preflight",return_value={"new_api_calls":0}) as pre, \
             patch("code_rsi.v3.calibration.generate") as generate, \
             patch("code_rsi.v3.calibration.credential_from_plan") as key, \
             contextlib.redirect_stdout(io.StringIO()):
            result=main(["preflight","--plan","synthetic.json"])
        self.assertEqual(result["new_api_calls"],0)
        pre.assert_called_once();generate.assert_not_called();key.assert_not_called()

    def test_generation_requires_hash_and_execute_flag(self):
        for args in (["calibrate","--plan","p"], ["calibrate","--plan","p","--execute"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_grade_follows_successful_generation_only(self):
        events=[]
        with patch("code_rsi.v3.calibration.load_plan",return_value=self.plan()), \
             patch("code_rsi.v3.calibration.generate",side_effect=lambda *a,**k:events.append(k)), \
             patch("code_rsi.v3.calibration.grade",side_effect=lambda p:events.append("grade") or {}), \
             contextlib.redirect_stdout(io.StringIO()):
            main(["calibrate","--plan","p","--execute","--approved-plan-hash","abc"])
        self.assertEqual(events,[{"approved_plan_hash":"abc"},"grade"])

    def test_generation_failure_does_not_grade(self):
        with patch("code_rsi.v3.calibration.load_plan",return_value=self.plan()), \
             patch("code_rsi.v3.calibration.generate",side_effect=RuntimeError("unknown outcome")), \
             patch("code_rsi.v3.calibration.grade") as grade, self.assertRaises(RuntimeError):
            main(["calibrate","--plan","p","--execute","--approved-plan-hash","abc"])
        grade.assert_not_called()

    def test_rejects_output_outside_runs(self):
        with patch("code_rsi.v3.calibration.load_plan",return_value={"output_dir":str(Path(__file__).parent)}), \
             patch("code_rsi.v3.calibration.preflight") as pre, self.assertRaises(ValueError):
            main(["preflight","--plan","p"])
        pre.assert_not_called()

if __name__=="__main__": unittest.main()
