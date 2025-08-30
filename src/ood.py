import os
import re
import json
import asyncio
from dotenv import load_dotenv
from datasets import load_dataset
from openai import AsyncOpenAI
import requests
from pathlib import Path
load_dotenv()

class out_of_distribution():

    def __init__(self, model_name, sample_size, at_pass = 1):
        self.model_name = model_name
        self.sample_size = sample_size
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
        self.at_pass = at_pass

        # datasets
        self.coin_flip = None
        self.lastletter = None

        # prompts
        self.system_prompt = self.load_prompt_template("prompts/system.txt")
        self.coin_flip_cot_prompt = self.load_prompt_template("prompts/ood/cot/cot_coinflip.txt")
        self.coin_flip_standard_prompt = self.load_prompt_template("prompts/ood/sp/sp_coinflip.txt")
        self.lastletter_cot_prompt = self.load_prompt_template("prompts/ood/cot/cot_lastletter.txt")
        self.lastletter_standard_prompt = self.load_prompt_template("prompts/ood/sp/sp_lastletter.txt")

        # concurrency limits
        self.semaphore = asyncio.Semaphore(2)

        # Centralized file naming convention
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.cot_filename_template = os.path.join(self.results_dir, "{model_name}_{dataset_name}_OOD_cot_@_pass{at_pass}.json")
        self.sp_filename_template = os.path.join(self.results_dir, "{model_name}_{dataset_name}_OOD_standard_@_pass{at_pass}.json")

    def load_prompt_template(self, file_path: str) -> str:
        """
        Load a text file containing placeholders `{}` as a string
        that can later be formatted with `.format()` or f-strings.
        """
        try:
            text = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            # If file doesn't exist or can't be read, return a simple default
            default = "Please answer the following question concisely:\n{Question}\nThe answer is"
            print(f"Warning: could not read prompt template {file_path}; using fallback template.")
            return default

        # If the prompt file exists but is empty, provide a safe default so prompts aren't blank
        if not text or not text.strip():
            default = "Please answer the following question concisely:\n{Question}\nThe answer is"
            print(f"Warning: prompt template {file_path} is empty; using fallback template.")
            return default

        return text

    def load_dataset(self, name):
        # Map canonical names to attribute names
        name_map = {
            "coin_flip": "coin_flip",
            "lastletter": "lastletter"
        }
        attr = name_map.get(name)
        if not attr:
            raise ValueError(f"Unknown dataset name: {name}")
        if getattr(self, attr) is not None:
            return getattr(self, attr)
        try:
            if name == "coin_flip":
                self.coin_flip = load_dataset("skrishna/coin_flip")
            elif name == "lastletter":
                self.lastletter = load_dataset("bythyag/LastLetterConcat-MultiNames")
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
                    ]
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

    def print_output(self, question, cot_model_output, sp_model_output, gold_answer,
                     cot_model_answer, sp_model_answer):

        print(f"QUESTION: \n{question}")
        print(f"==========ANSWER SUMMARY============")
        print(f"Gold Answer: {gold_answer}")
        print(f"CoT Answer: {cot_model_answer}")
        print(f"Standard Prompt Answer: {sp_model_answer}")
        print("="*60)

        # Evaluate CoT
        if cot_model_answer == gold_answer:
            cot_result = "Correct"
        else:
            cot_result = "Incorrect"

        # Evaluate Standard
        if sp_model_answer == gold_answer:
            sp_result = "Correct"
        else:
            sp_result = "Incorrect"

        print("="*60)
        return cot_result, sp_result

    def store_results(self, cot_results, sp_results, question, cot_model_output, sp_model_output, cot_model_answer, sp_model_answer, gold_answer, original_answer, cot_result, sp_result, cot_prompt, sp_prompt):
        cot_results.append({
            "question": question,
            "prompt_used": cot_prompt,
            "model_output": cot_model_output,
            "extracted_answer": cot_model_answer,
            "gold_answer": gold_answer,
            "result": cot_result,
            "@ pass": self.at_pass
        })
        sp_results.append({
            "question": question,
            "prompt_used": sp_prompt,
            "model_output": sp_model_output,
            "extracted_answer": sp_model_answer,
            "gold_answer": gold_answer,
            "result": sp_result,
            "@ pass": self.at_pass
        })
        return cot_results, sp_results

    def print_final_score(self, cot_correct_count, cot_incorrect_count, cot_na_count,
                          sp_correct_count, sp_incorrect_count, sp_na_count):

        print(f"=====FINAL SCORE (Estimated) for {self.model_name} =====")
        print("Chain-of-Thought Results:")
        print(f"  Correct   : {cot_correct_count}")
        # compute denominator from actual processed counts to avoid misleading percentages
        cot_denom = cot_correct_count + cot_incorrect_count + cot_na_count
        if cot_denom == 0:
            cot_denom = self.sample_size
        cot_accuracy = (cot_correct_count / cot_denom) if cot_denom else 0
        print(f"  Accuracy  : {cot_correct_count}/{cot_denom} = {cot_accuracy:.2%}")
        print("-"*60)
        print("Standard Prompting Results:")
        print(f"  Correct   : {sp_correct_count}")
        sp_denom = sp_correct_count + sp_incorrect_count + sp_na_count
        if sp_denom == 0:
            sp_denom = self.sample_size
        sp_accuracy = (sp_correct_count / sp_denom) if sp_denom else 0
        print(f"  Accuracy  : {sp_correct_count}/{sp_denom} = {sp_accuracy:.2%}")
        print("Note: These results are based on extracted regex patterns. Some answers may escape the detection. Please verify once manually for all the incorrect answers.")
        print("="*60)

    async def handle_question(self, question, original_answer, cot_prompt, sp_prompt, semaphore):
        # run both prompts concurrently
        cot_task = asyncio.create_task(self.model_runner(self.model_name, self.system_prompt, cot_prompt, semaphore))
        sp_task = asyncio.create_task(self.model_runner(self.model_name, self.system_prompt, sp_prompt, semaphore))
        cot_model_output, sp_model_output = await asyncio.gather(cot_task, sp_task)

        gold_answer = self.extract_after(r"####\s*(.*)", original_answer)
        cot_model_answer = self.extract_after(r"The answer is\s*(.*)", cot_model_output or "")
        sp_model_answer = self.extract_after(r"The answer is\s*(.*)", sp_model_output or "")

        cot_result, sp_result = self.print_output(
            question, cot_model_output, sp_model_output, gold_answer, cot_model_answer, sp_model_answer
        )

        # Modify this line to pass the prompts
        cot_results, sp_results = self.store_results(
            [], [], question, cot_model_output, sp_model_output, cot_model_answer, sp_model_answer, gold_answer, original_answer, cot_result, sp_result, cot_prompt, sp_prompt
        )

        return (cot_results, sp_results, cot_result, sp_result)

    def aggregate_results(self, results, dataset_name):
        cot_results, sp_results = [], []
        cot_correct_count = cot_incorrect_count = cot_na_count = 0
        sp_correct_count = sp_incorrect_count = sp_na_count = 0

        for cot_res, sp_res, cot_status, sp_status in results:
            cot_results.extend(cot_res)
            sp_results.extend(sp_res)

            if cot_status == "Correct":
                cot_correct_count += 1
            elif cot_status == "Incorrect":
                cot_incorrect_count += 1
            else:
                cot_na_count += 1

            if sp_status == "Correct":
                sp_correct_count += 1
            elif sp_status == "Incorrect":
                sp_incorrect_count += 1
            else:
                sp_na_count += 1

        # Save results to JSON files using centralized naming
        cot_filename = self.cot_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        sp_filename = self.sp_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)

        # self.save_results_to_json(cot_results, cot_filename)
        # self.save_results_to_json(sp_results, sp_filename)

        # Print final scores
        self.print_final_score(
            cot_correct_count, cot_incorrect_count, cot_na_count,
            sp_correct_count, sp_incorrect_count, sp_na_count
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
        Generic runner. `question_factory(i)` -> (question, original_answer, cot_prompt, sp_prompt).
        """
        # ensure dataset loaded
        if getattr(self, dataset_name) is None:
            self.load_dataset(dataset_name)

        # filenames / processed tracking
        cot_filename = self.cot_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)
        sp_filename = self.sp_filename_template.format(model_name=self.model_name, dataset_name=dataset_name, at_pass=self.at_pass)

        # Ensure files exist and are initialized as empty lists if not
        for filename in [cot_filename, sp_filename]:
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump([], f)

        processed_questions = self.load_processed_questions(cot_filename).union(self.load_processed_questions(sp_filename))
        semaphore = self.semaphore

        for i in range(self.sample_size):
            try:
                question, original_answer, cot_prompt, sp_prompt = question_factory(i)
            except Exception as e:
                print(f"[{dataset_name}] skipping index {i}: factory error: {e}")
                continue

            if question in processed_questions:
                print(f"Skipping question: {question} (already processed)")
                continue

            # process question and immediately append results to file
            cot_results, sp_results, cot_result, sp_result = await self.handle_question(question, original_answer, cot_prompt, sp_prompt, semaphore)

            # Append results to JSON files as we go
            self.save_results_to_json(cot_results, cot_filename)
            self.save_results_to_json(sp_results, sp_filename)

            # Update processed_questions set
            processed_questions.add(question)

        print(f"Finished processing {dataset_name}. Results saved to {cot_filename} and {sp_filename}.")
        # Aggregate and print final scores
        all_results = []
        try:
            with open(cot_filename, "r", encoding="utf-8") as f_cot, open(sp_filename, "r", encoding="utf-8") as f_sp:
                cot_data = json.load(f_cot)
                sp_data = json.load(f_sp)
                # Pair results by question (assuming same order)
                for cot_item, sp_item in zip(cot_data, sp_data):
                    cot_status = cot_item.get("result", "NA")
                    sp_status = sp_item.get("result", "NA")
                    all_results.append(([cot_item], [sp_item], cot_status, sp_status))
        except Exception as e:
            print(f"Error aggregating results for final score: {e}")
            return
        self.aggregate_results(all_results, dataset_name)

    async def run_coin_flip(self):
        coin_flip = self.load_dataset("coin_flip")
        if coin_flip is None:
            print("coin_flip dataset could not be loaded.")
            return
        await self.run_dataset("coin_flip", lambda i: (
            (lambda item: (
                item['inputs'],
                item['targets'],
                self.coin_flip_cot_prompt.format(Question=item['inputs']),
                self.coin_flip_standard_prompt.format(Question=item['inputs'])
            ))(coin_flip['test'][i])
        ))

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
            q = ex.get('question', ex.get('inputs'))
            ans = ex.get('answer', ex.get('targets'))
            return (
                q,
                ans,
                self.lastletter_cot_prompt.format(Question=q),
                self.lastletter_standard_prompt.format(Question=q),
            )
        await self.run_dataset("lastletter", factory)

    async def run_all(self):
        """Run benchmark on all OOD datasets"""
        datasets = [
            ("coin_flip", self.run_coin_flip),
            ("lastletter", self.run_lastletter)
        ]

        print(f"Starting OOD benchmark on {len(datasets)} datasets with {self.sample_size} samples each...")
        print(f"Model: {self.model_name}")
        print(f"Pass: {self.at_pass}")
        print("="*60)

        for dataset_name, run_func in datasets:
            try:
                print(f"\n🔄 Running OOD benchmark on {dataset_name.upper()}...")
                await run_func()
                print(f"✅ Completed {dataset_name}")
            except Exception as e:
                print(f"❌ Error running {dataset_name}: {str(e)}")
                continue

        print("\n" + "="*60)
        print("OOD Benchmark completed for all datasets!")

# Add a main function for CLI usage
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run OOD benchmarks on specific datasets.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gpt-4o)')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of samples per dataset')
    parser.add_argument('--at_pass', type=int, default=1, help='Pass number')
    parser.add_argument('--run', type=str, default='all', choices=[
        'all', 'coin_flip', 'lastletter'
    ], help='Which OOD benchmark to run')
    args = parser.parse_args()

    ood_bench = out_of_distribution(args.model_name, args.sample_size, args.at_pass)

    async def runner():
        if args.run == 'all':
            await ood_bench.run_all()
        else:
            func = getattr(ood_bench, f"run_{args.run}", None)
            if func is None:
                print(f"No such OOD benchmark: {args.run}")
                return
            await func()

    import asyncio
    asyncio.run(runner())

if __name__ == "__main__":
    main()

# python src/ood.py --model_name gpt-4o-mini --at_pass 1 --sample_size 5 --run coin_flip
