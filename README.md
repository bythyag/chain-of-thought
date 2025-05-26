# replication experiment on chain-of-thought reasoning

paper link: [chain-of-thought prompting elicits reasoning in large language models](https://arxiv.org/pdf/2201.11903)

## paper summary

llms are capable of a multitude of language understanding tasks such as question answering, analysis, translation, and information extraction. however, problems in logical reasoning and arithmetic require multiple reasoning steps to solve correctly. for such problem categories, models like gpt-3 and palm exhibit low accuracy scores. while these problems can be addressed through fine-tuning for specific tasks, creating datasets for such cases is tedious, costly, and time-consuming.

to address this, the paper proposes solving these problems using prompting methods. it introduces chain of thought (cot) prompting, which improves poor reasoning performance in language models by:

- generating a series of intermediate reasoning steps before arriving at the final solution,
- using few-shot examples that include reasoning steps to guide the model,
- formatting prompts as triples of `<input, chain of thought, output>` for reference.

cot prompting demonstrates significant performance improvements across arithmetic, commonsense, and symbolic reasoning tasks.

## why does it work?

cot prompting enables the model to decompose multi-step problems into intermediate steps. it also provides an interpretable reasoning path, which is useful for understanding the model’s logic and debugging incorrect answers.

## example prompt

prompt:
```python
question 1:
roger has 5 tennis balls. he buys 2 more cans of tennis balls. each can has 3 tennis balls.
how many tennis balls does he have now?

answer:
roger started with 5 balls.
2 cans × 3 tennis balls = 6 balls.
5 + 6 = 11
the answer is 11.

question 2:
the cafeteria had 23 apples. they used 20 to make lunch and bought 6 more.
how many apples do they have now?
```

llm response:
```bash
the cafeteria had 23 apples originally.
they used 20: 23 − 20 = 3 apples left.
they bought 6 more: 3 + 6 = 9
the answer is 9.
```


in collaboration with [@telt](https://github.com/teltam)
