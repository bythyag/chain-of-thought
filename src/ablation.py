import os
import re
import json
import asyncio
import requests
from datasets import load_dataset
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Ablation:
    def __init__(self, model_name, sample_size, at_pass=1):
        self.model_name = model_name
        self.sample_size = sample_size
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
        self.at_pass = at_pass

        # datasets
        self.gsm8k = None
        self.svamp = None
        self.asdiv = None
        self.aqua = None
        self.mawps = None
        self.csqa = None
        self.strategyqa = None
        self.date_bench = None
        self.sports_bench = None
        self.saycan = None
        self.coin_flip = None
        self.lastletter = None

        # prompts - load from files (these are the system prompts and templates for the three ablation studies)
        self.prompt_equation_only = self.load_prompt_template("prompts/ablation/equation_only.txt")
        self.prompt_variable_compute = self.load_prompt_template("prompts/ablation/variable_compute.txt")
        self.prompt_variable_compute_template = self.load_prompt_template("prompts/ablation/variable_compute_template.txt")
        self.prompt_post_answer_cot = self.load_prompt_template("prompts/ablation/reasoning_post_answer.txt")

        # concurrency limits - initialize lazily to avoid event loop issues
        self._semaphore = None

        # Centralized file naming convention
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.eq_filename_template = os.path.join(self.results_dir, "{model_name}_{dataset_name}_equation_only_@_pass{at_pass}.json")
        self.var_filename_template = os.path.join(self.results_dir, "{model_name}_{dataset_name}_variable_compute_@_pass{at_pass}.json")
        self.cot_filename_template = os.path.join(self.results_dir, "{model_name}_{dataset_name}_post_answer_cot_@_pass{at_pass}.json")

    @property
    def semaphore(self):
        """Lazy initialization of semaphore to avoid event loop issues"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(2)
        return self._semaphore

    def load_prompt_template(self, file_path: str) -> str:
        """
        Load a text file containing placeholders `{}` as a string
        that can later be formatted with `.format()` or f-strings.
        """
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            # If file doesn't exist or can't be read, return a simple default
            default = "You are a helpful assistant. Please answer the following question:\n{Question}\nThe answer is"
            print(f"Warning: could not read prompt template {file_path}; using fallback template.")
            return default

        # If the prompt file exists but is empty, provide a safe default so prompts aren't blank
        if not text or not text.strip():
            default = "You are a helpful assistant. Please answer the following question:\n{Question}\nThe answer is"
            print(f"Warning: prompt template {file_path} is empty; using fallback template.")
            return default

        return text

    def load_dataset(self, name):
        # Map canonical names to attribute names
        name_map = {
            "gsm8k": "gsm8k",
            "svamp": "svamp",
            "asdiv": "asdiv",
            "aqua": "aqua",
            "mawps": "mawps",
            "csqa": "csqa",
            "strategyqa": "strategyqa",
            "date_bench": "date_bench",
            "sports_bench": "sports_bench",
            "saycan": "saycan",
            "lastletter": "lastletter",
            "coin_flip": "coin_flip"
        }
        attr = name_map.get(name)
        if not attr:
            raise ValueError(f"Unknown dataset name: {name}")
        if getattr(self, attr) is not None:
            return getattr(self, attr)
        try:
            if name == "gsm8k":
                self.gsm8k = load_dataset("openai/gsm8k", "main")
            elif name == "svamp":
                self.svamp = load_dataset("ChilleD/SVAMP")
            elif name == "asdiv":
                self.asdiv = load_dataset("yimingzhang/asdiv")
            elif name == "aqua":
                self.aqua = load_dataset("deepmind/aqua_rat")
            elif name == "mawps":
                self.mawps = load_dataset("MU-NLPC/Calc-mawps")
            elif name == "csqa":
                self.csqa = load_dataset("commonsense_qa")
            elif name == "strategyqa":
                self.strategyqa = load_dataset("metaeval/strategy-qa")
            elif name == "date_bench":
                date_url = "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json"
                date_response = requests.get(date_url)
                self.date_bench = json.loads(date_response.text)
            elif name == "sports_bench":
                sports_url = "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/sports_understanding/task.json"
                sports_response = requests.get(sports_url)
                self.sports_bench = json.loads(sports_response.text)
            elif name == "saycan":
                self.saycan = load_dataset("chiayewken/saycan")
            elif name == "lastletter":
                self.lastletter = load_dataset("bythyag/LastLetterConcat-MultiNames")
            elif name == "coin_flip":
                self.coin_flip = load_dataset("skrishna/coin_flip")
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
            setattr(self, attr, None)
        return getattr(self, attr)

    async def model_runner(self, model_name, system_message, prompt, semaphore):
        async with semaphore:  # limit concurrency
            if "gpt" in model_name:
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                return response.choices[0].message.content
            if "test" in model_name:
                return "The answer is test"
            return None

    def save_results_to_json(self, results, filename):
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []
        existing_data.extend(results)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

    def extract_after(self, pattern, text, fallback=True):
        match = re.search(pattern, text)
        if match:
            extracted = match.group(1).strip().rstrip('.')
        else:
            if not fallback:
                return None
            extracted = text.strip().rstrip('.')

        # Normalize numbers: remove commas, convert floats like 23.00 → 23
        if re.fullmatch(r"[\d,]+(\.\d+)?", extracted):
            extracted = extracted.replace(",", "")
            try:
                num = float(extracted)
                if num.is_integer():
                    extracted = str(int(num))
                else:
                    extracted = str(num)
            except ValueError:
                pass

        return extracted

    def print_output(self, question, eq_output, var_output, cot_output, gold_answer,
                     eq_answer, var_answer, cot_answer):

        print(f"QUESTION: \n{question}")
        print(f"==========ANSWER SUMMARY============")
        print(f"Gold Answer: {gold_answer}")
        print(f"Equation Only Answer: {eq_answer}")
        print(f"Variable Compute Answer: {var_answer}")
        print(f"Post Answer CoT Answer: {cot_answer}")
        print("="*60)

        # Evaluate each ablation type
        eq_result = "Correct" if eq_answer == gold_answer else "Incorrect"
        var_result = "Correct" if var_answer == gold_answer else "Incorrect"
        cot_result = "Correct" if cot_answer == gold_answer else "Incorrect"

        print("="*60)
        return eq_result, var_result, cot_result

    def store_results(self, eq_results, var_results, cot_results, question, 
                     eq_output, var_output, cot_output, eq_answer, var_answer, cot_answer, 
                     gold_answer, original_answer, eq_result, var_result, cot_result,
                     eq_prompt, var_prompt, cot_prompt):
        
        eq_results.append({
            "question": question,
            "prompt_used": eq_prompt,
            "model_output": eq_output,
            "extracted_answer": eq_answer,
            "gold_answer": gold_answer,
            "result": eq_result,
            "@ pass": self.at_pass
        })
        
        var_results.append({
            "question": question,
            "prompt_used": var_prompt,
            "model_output": var_output,
            "extracted_answer": var_answer,
            "gold_answer": gold_answer,
            "result": var_result,
            "@ pass": self.at_pass
        })
        
        cot_results.append({
            "question": question,
            "prompt_used": cot_prompt,
            "model_output": cot_output,
            "extracted_answer": cot_answer,
            "gold_answer": gold_answer,
            "result": cot_result,
            "@ pass": self.at_pass
        })
        
        return eq_results, var_results, cot_results

    def print_final_score(self, eq_correct, eq_incorrect, eq_na,
                          var_correct, var_incorrect, var_na,
                          cot_correct, cot_incorrect, cot_na):

        print(f"=====FINAL SCORE (Estimated) for {self.model_name} =====")
        
        print("Equation Only Results:")
        print(f"  Correct   : {eq_correct}")
        eq_denom = eq_correct + eq_incorrect + eq_na
        if eq_denom == 0:
            eq_denom = self.sample_size
        eq_accuracy = (eq_correct / eq_denom) if eq_denom else 0
        print(f"  Accuracy  : {eq_correct}/{eq_denom} = {eq_accuracy:.2%}")
        print("-"*60)
        
        print("Variable Compute Results:")
        print(f"  Correct   : {var_correct}")
        var_denom = var_correct + var_incorrect + var_na
        if var_denom == 0:
            var_denom = self.sample_size
        var_accuracy = (var_correct / var_denom) if var_denom else 0
        print(f"  Accuracy  : {var_correct}/{var_denom} = {var_accuracy:.2%}")
        print("-"*60)
        
        print("Post Answer CoT Results:")
        print(f"  Correct   : {cot_correct}")
        cot_denom = cot_correct + cot_incorrect + cot_na
        if cot_denom == 0:
            cot_denom = self.sample_size
        cot_accuracy = (cot_correct / cot_denom) if cot_denom else 0
        print(f"  Accuracy  : {cot_correct}/{cot_denom} = {cot_accuracy:.2%}")
        
        print("Note: These results are based on extracted regex patterns. Some answers may escape the detection. Please verify once manually for all the incorrect answers.")
        print("="*60)

    async def handle_question(self, question, original_answer, eq_prompt, var_prompt, cot_prompt, semaphore):
        # run all three prompts concurrently with their respective system prompts
        eq_task = asyncio.create_task(self.model_runner(self.model_name, self.prompt_equation_only, eq_prompt, semaphore))
        # Format variable compute prompt using the template
        var_templated_prompt = self.prompt_variable_compute_template.format(Question=var_prompt)
        var_task = asyncio.create_task(self.model_runner(self.model_name, self.prompt_variable_compute, var_templated_prompt, semaphore))
        cot_task = asyncio.create_task(self.model_runner(self.model_name, self.prompt_post_answer_cot, cot_prompt, semaphore))
        
        eq_output, var_output, cot_output = await asyncio.gather(eq_task, var_task, cot_task)

        gold_answer = self.extract_after(r"####\s*(.*)", original_answer)
        eq_answer = self.extract_after(r"The answer is\s*(.*)", eq_output or "")
        var_answer = self.extract_after(r"The answer is\s*(.*)", var_output or "")
        cot_answer = self.extract_after(r"The answer is\s*(.*)", cot_output or "")

        eq_result, var_result, cot_result = self.print_output(
            question, eq_output, var_output, cot_output, gold_answer, 
            eq_answer, var_answer, cot_answer
        )

        eq_results, var_results, cot_results = self.store_results(
            [], [], [], question, eq_output, var_output, cot_output, 
            eq_answer, var_answer, cot_answer, gold_answer, original_answer, 
            eq_result, var_result, cot_result, eq_prompt, var_prompt, cot_prompt
        )

        return (eq_results, var_results, cot_results, eq_result, var_result, cot_result)

    def aggregate_results(self, results, dataset_name):
        eq_results, var_results, cot_results = [], [], []
        eq_correct = eq_incorrect = eq_na = 0
        var_correct = var_incorrect = var_na = 0
        cot_correct = cot_incorrect = cot_na = 0

        for eq_res, var_res, cot_res, eq_status, var_status, cot_status in results:
            eq_results.extend(eq_res)
            var_results.extend(var_res)
            cot_results.extend(cot_res)

            if eq_status == "Correct":
                eq_correct += 1
            elif eq_status == "Incorrect":
                eq_incorrect += 1
            else:
                eq_na += 1

            if var_status == "Correct":
                var_correct += 1
            elif var_status == "Incorrect":
                var_incorrect += 1
            else:
                var_na += 1

            if cot_status == "Correct":
                cot_correct += 1
            elif cot_status == "Incorrect":
                cot_incorrect += 1
            else:
                cot_na += 1

        # Save results to JSON files using centralized naming
        eq_filename = self.eq_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        var_filename = self.var_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        cot_filename = self.cot_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)

        # Print final scores
        self.print_final_score(
            eq_correct, eq_incorrect, eq_na,
            var_correct, var_incorrect, var_na,
            cot_correct, cot_incorrect, cot_na
        )

    def load_processed_questions(self, filename):
        """Loads a set of questions from an existing results JSON file."""
        if not os.path.exists(filename):
            return set()

        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return {item['question'] for item in data}
            except json.JSONDecodeError:
                return set()

    async def run_dataset(self, dataset_name: str, question_factory):
        """
        Generic runner. `question_factory(i)` -> (question, original_answer, eq_prompt, var_prompt, cot_prompt).
        """
        # ensure dataset loaded
        if getattr(self, dataset_name) is None:
            self.load_dataset(dataset_name)

        # filenames / processed tracking
        eq_filename = self.eq_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        var_filename = self.var_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        cot_filename = self.cot_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)

        # Ensure files exist and are initialized as empty lists if not
        for filename in [eq_filename, var_filename, cot_filename]:
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump([], f)

        processed_questions = (self.load_processed_questions(eq_filename)
                             .union(self.load_processed_questions(var_filename))
                             .union(self.load_processed_questions(cot_filename)))
        semaphore = self.semaphore

        for i in range(self.sample_size):
            try:
                question, original_answer, eq_prompt, var_prompt, cot_prompt = question_factory(i)
            except Exception as e:
                print(f"[{dataset_name}] skipping index {i}: factory error: {e}")
                continue

            if question in processed_questions:
                print(f"Skipping question: {question} (already processed)")
                continue

            # process question and immediately append results to file
            eq_results, var_results, cot_results, eq_result, var_result, cot_result = await self.handle_question(
                question, original_answer, eq_prompt, var_prompt, cot_prompt, semaphore
            )

            # Append results to JSON files as we go
            self.save_results_to_json(eq_results, eq_filename)
            self.save_results_to_json(var_results, var_filename)
            self.save_results_to_json(cot_results, cot_filename)

            # Update processed_questions set
            processed_questions.add(question)

        print(f"Finished processing {dataset_name}. Results saved to {eq_filename}, {var_filename}, and {cot_filename}.")
        
        # Aggregate and print final scores
        all_results = []
        try:
            with open(eq_filename, "r", encoding="utf-8") as f_eq, \
                 open(var_filename, "r", encoding="utf-8") as f_var, \
                 open(cot_filename, "r", encoding="utf-8") as f_cot:
                eq_data = json.load(f_eq)
                var_data = json.load(f_var)
                cot_data = json.load(f_cot)
                # Pair results by question (assuming same order)
                for eq_item, var_item, cot_item in zip(eq_data, var_data, cot_data):
                    eq_status = eq_item.get("result", "NA")
                    var_status = var_item.get("result", "NA")
                    cot_status = cot_item.get("result", "NA")
                    all_results.append(([eq_item], [var_item], [cot_item], eq_status, var_status, cot_status))
        except Exception as e:
            print(f"Error aggregating results for final score: {e}")
            return
        self.aggregate_results(all_results, dataset_name)

    async def run_gsm8k(self):
        gsm8k = self.load_dataset("gsm8k")
        if gsm8k is None:
            print("gsm8k dataset could not be loaded.")
            return
        await self.run_dataset("gsm8k", lambda i: (
            (lambda item: (
                item['question'],
                item['answer'],
                item['question'],  # eq_prompt (just the question)
                item['question'],  # var_prompt (just the question)
                item['question']   # cot_prompt (just the question)
            ))(gsm8k['test'][i])
        ))

    async def run_svamp(self):
        svamp = self.load_dataset("svamp")
        if svamp is None:
            print("svamp dataset could not be loaded.")
            return
        await self.run_dataset("svamp", lambda i: (
            (lambda item: (
                item['question_concat'],
                f"#### {item['Answer']}",
                item['question_concat'],
                item['question_concat'],
                item['question_concat']
            ))(svamp['train'][i])
        ))

    async def run_mawps(self):
        mawps = self.load_dataset("mawps")
        if mawps is None:
            print("mawps dataset could not be loaded.")
            return
        await self.run_dataset("mawps", lambda i: (
            (lambda item: (
                item['question'],
                f"#### {item['result']}",
                item['question'],
                item['question'],
                item['question']
            ))(mawps['train'][i])
        ))

    async def run_asdiv(self):
        asdiv = self.load_dataset("asdiv")
        if asdiv is None:
            print("asdiv dataset could not be loaded.")
            return
        def factory(i):
            raw = str(asdiv['train'][i]['text'])
            q = re.sub(r'^Question: |\nAnswer:$', '', raw).strip()
            return q, f"#### {asdiv['train'][i]['label']}", q, q, q
        await self.run_dataset("asdiv", factory)

    async def run_aqua(self):
        aqua = self.load_dataset("aqua")
        if aqua is None:
            print("aqua dataset could not be loaded.")
            return
        def factory(i):
            item = aqua['train'][i]
            formatted_options = "Answer Choices: " + " ".join(
                f"({chr(97+j)}) {str(opt)[2:]}" for j, opt in enumerate(item['options'])
            )
            q = f"{item['question']}\n{formatted_options}"
            return q, f"#### {str(item['correct']).lower()}", q, q, q
        await self.run_dataset("aqua", factory)

    async def run_csqa(self):
        csqa = self.load_dataset("csqa")
        if csqa is None:
            print("csqa dataset could not be loaded.")
            return
        def factory(i):
            ex = csqa['train'][i]
            labels = [str(label).lower() for label in ex['choices']['label']]
            texts = ex['choices']['text']
            choices_str = " ".join([f"({label}) {text}" for label, text in zip(labels, texts)])
            q = f"{ex['question']} Answer Choices: {choices_str}"
            return q, f"#### {str(ex['answerKey']).lower()}", q, q, q
        await self.run_dataset("csqa", factory)

    async def run_strategyqa(self):
        strategyqa = self.load_dataset("strategyqa")
        if strategyqa is None:
            print("strategyqa dataset could not be loaded.")
            return
        def factory(i):
            q = str(strategyqa['train'][i]['question'])
            ans = "yes" if strategyqa['train'][i]['answer'] else "no"
            return q, f"#### {ans}", q, q, q
        await self.run_dataset("strategyqa", factory)

    async def run_date_bench(self):
        date_bench = self.load_dataset("date_bench")
        if date_bench is None:
            print("date_bench dataset could not be loaded.")
            return
        def factory(i):
            ex = date_bench['examples'][i]
            question = str(ex['input'])
            original_answer = str(next((k for k, v in ex['target_scores'].items() if v == 1), ""))
            return question, f"#### {original_answer}", question, question, question
        await self.run_dataset("date_bench", factory)

    async def run_sports_bench(self):
        sports_bench = self.load_dataset("sports_bench")
        if sports_bench is None:
            print("sports_bench dataset could not be loaded.")
            return
        def factory(i):
            ex = sports_bench['examples'][i]
            question = f'Is the following sentence plausible? "{ex["input"]}"'
            original_answer = "yes" if ex["target_scores"].get("plausible", 0) == 1 else "no"
            return question, f"#### {original_answer}", question, question, question
        await self.run_dataset("sports_bench", factory)

    async def run_saycan(self):
        saycan = self.load_dataset("saycan")
        if saycan is None:
            print("saycan dataset could not be loaded.")
            return
        def factory(i):
            q = str(saycan['test'][i]['INPUT'])
            ans = str(saycan['test'][i]['OUTPUT'])
            return q, f"#### {ans}", q, q, q
        await self.run_dataset("saycan", factory)

    async def run_lastletter(self):
        lastletter = self.load_dataset("lastletter")
        if lastletter is None:
            print("lastletter dataset could not be loaded.")
            return
        half = self.sample_size // 2
        total_3 = len(lastletter['3_names'])
        total_4 = len(lastletter['4_names'])
        def factory(i):
            if i < min(half, total_3):
                ex = lastletter['3_names'][i]
            else:
                idx = i - min(half, total_3)
                if idx < total_4:
                    ex = lastletter['4_names'][idx]
                else:
                    raise IndexError("Index out of range for lastletter")
            q = str(ex.get('question', ex.get('inputs', '')))
            ans = str(ex.get('answer', ex.get('targets', '')))
            return q, f"#### {ans}", q, q, q
        await self.run_dataset("lastletter", factory)

    async def run_coin_flip(self):
        coinflip = self.load_dataset("coin_flip")
        if coinflip is None:
            print("coin_flip dataset could not be loaded.")
            return
        def factory(i):
            try:
                ex = coinflip['test'][i]
                q = str(ex['inputs'])
                ans = str(ex['targets'])
            except (KeyError, IndexError):
                # Try alternative structure
                ex = coinflip['train'][i]
                q = str(ex.get('inputs', ex.get('question', '')))
                ans = str(ex.get('targets', ex.get('answer', '')))
            
            q = q.lstrip("Q: ").strip()
            return q, f"#### {ans}", q, q, q
        await self.run_dataset("coin_flip", factory)

    async def run_all(self):
        """Run ablation experiments on all datasets"""
        datasets = [
            ("gsm8k", self.run_gsm8k),
            ("svamp", self.run_svamp),
            ("mawps", self.run_mawps),
            ("asdiv", self.run_asdiv),
            ("aqua", self.run_aqua),
            ("csqa", self.run_csqa),
            ("strategyqa", self.run_strategyqa),
            ("date_bench", self.run_date_bench),
            ("sports_bench", self.run_sports_bench),
            ("saycan", self.run_saycan),
            ("lastletter", self.run_lastletter),
            ("coin_flip", self.run_coin_flip)
        ]

        print(f"Starting ablation experiments on {len(datasets)} datasets with {self.sample_size} samples each...")
        print(f"Model: {self.model_name}")
        print(f"Pass: {self.at_pass}")
        print("="*60)

        for dataset_name, run_func in datasets:
            try:
                print(f"\n🔄 Running ablation on {dataset_name.upper()}...")
                await run_func()
                print(f"✅ Completed {dataset_name}")
            except Exception as e:
                print(f"❌ Error running {dataset_name}: {str(e)}")
                continue

        print("\n" + "="*60)
        print("Ablation experiments completed for all datasets!")

# Add a main function for CLI usage
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation experiments on various datasets.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gpt-4o-mini)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples per dataset')
    parser.add_argument('--at_pass', type=int, default=1, help='Pass number')
    parser.add_argument('--run', type=str, default='all', choices=[
        'all', 'gsm8k', 'svamp', 'mawps', 'asdiv', 'aqua', 'csqa', 'strategyqa', 
        'date_bench', 'sports_bench', 'saycan', 'lastletter', 'coin_flip'
    ], help='Which ablation experiment to run')
    args = parser.parse_args()

    ablation = Ablation(args.model_name, args.sample_size, args.at_pass)

    async def runner():
        if args.run == 'all':
            await ablation.run_all()
        else:
            func = getattr(ablation, f"run_{args.run}", None)
            if func is None:
                print(f"No such ablation experiment: {args.run}")
                return
            await func()

    import asyncio
    asyncio.run(runner())

if __name__ == "__main__":
    main()

# Example usage:
# python src/ablation.py --model_name gpt-4o-mini --at_pass 1 --sample_size 5 --run gsm8k
