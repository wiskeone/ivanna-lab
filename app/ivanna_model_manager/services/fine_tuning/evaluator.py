import logging
import torch
import numpy as np
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import evaluate
from ...models import FineTuningJob, FineTuningEvaluation, EvaluationDataset
import traceback


logger = logging.getLogger(__name__)


class FineTuningEvaluator:
    """Evaluate a fine-tuned model on datasets"""

    def __init__(self, job: FineTuningJob):
        self.job = job
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the metrics
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load('bertscore')
        self.bleu = evaluate.load('bleu')
    
    def evaluate(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """Evaluate the model on a dataset"""
        logger.info(f"Evaluating model on dataset: {dataset.name}")

        # Load the model
        model_path = self.job.output_model.local_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        predictions = []
        references = []
        
        for q in dataset.questions[:50]:  # cap for speed
            # Audit fix: only 'answer' was accepted — datasets written with
            # 'output' (HF import / distillation export) were skipped entirely
            if 'answer' not in q and 'output' not in q:
                continue

            prompt = f"### Instruction:\n{q.get('text', q.get('instruction', ''))}\n\n### Response:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the answer only
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            predictions.append(response)
            references.append(q.get('answer', q.get('output', '')))
        
        # Compute metrics
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_aggregator=True
        )
        
        bert_scores = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="ru"  # or detect automatically
        )
        
        bleu_scores = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        
        metrics = {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bertscore_mean': np.mean(bert_scores['f1']),
            'bleu': bleu_scores['bleu'],
        }
        
        # Store the results
        evaluation = FineTuningEvaluation.objects.create(
            job=self.job,
            dataset=dataset,
            metrics=metrics,
            detailed_results=[
                {'prediction': p, 'reference': r} 
                for p, r in zip(predictions[:10], references[:10])  # first 10 only
            ]
        )
        
        logger.info(f"Evaluation completed: {metrics}")
        return metrics