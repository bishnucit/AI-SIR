import time
import json
from datetime import datetime
from ollama import chat
from tabulate import tabulate

LOG_FILE = "ollama_logs_normalized.jsonl"


MODELS = [
    "model1",
    "model2",
    "model3",
]


# Common settings for all models
# This makes the comparison fair
OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "num_predict": 300,
    "think": False
}


TEST_PROMPTS = [
    {
        "name": "Reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Explain briefly."
    },
    {
        "name": "Coding",
        "prompt": "Write a Python function to reverse a string. Keep it under 10 lines."
    },
    {
        "name": "Knowledge",
        "prompt": "Explain why the sky appears blue in exactly 3 sentences."
    },
    {
        "name": "Creativity",
        "prompt": "Write a short story about a robot discovering music in 5 sentences."
    },
    {
        "name": "Instruction Following",
        "prompt": "List exactly 5 benefits of solar energy using bullet points only."
    }
]


def log_call(model, category, prompt, response, elapsed):

    prompt_tokens = response.get("prompt_eval_count", 0)
    response_tokens = response.get("eval_count", 0)

    total_tokens = (
        prompt_tokens +
        response_tokens
    )

    generation_time = (
        response.get("eval_duration", 0) / 1e9
    )

    tps = (
        response_tokens / generation_time
        if generation_time > 0
        else 0
    )


    log = {

        "timestamp": datetime.now().isoformat(),

        "model": model,
        "category": category,

        "benchmark_settings": OPTIONS,

        "prompt": prompt,

        "response": response["message"]["content"],


        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,


        "latency_seconds": round(
            elapsed,
            3
        ),

        "tokens_per_second": round(
            tps,
            2
        ),


        "prompt_eval_duration_ns":
            response.get(
                "prompt_eval_duration"
            ),

        "eval_duration_ns":
            response.get(
                "eval_duration"
            ),

        "total_duration_ns":
            response.get(
                "total_duration"
            ),
    }


    with open(LOG_FILE, "a") as f:
        f.write(
            json.dumps(log) + "\n"
        )


    return log



def ask(model, category, prompt):

    print(
        f"\nRunning {model} - {category}"
    )


    start = time.perf_counter()


    response = chat(

        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        options=OPTIONS

    )


    elapsed = (
        time.perf_counter()
        -
        start
    )


    return log_call(
        model,
        category,
        prompt,
        response,
        elapsed
    )



def main():

    results = []


    for model in MODELS:

        for test in TEST_PROMPTS:

            try:

                stats = ask(
                    model,
                    test["name"],
                    test["prompt"]
                )

                results.append(stats)


            except Exception as e:

                print(
                    f"{model} failed on {test['name']}: {e}"
                )



    table = []


    for r in results:

        table.append([

            r["model"],

            r["category"],

            r["total_tokens"],

            f"{r['latency_seconds']:.2f}s",

            f"{r['tokens_per_second']:.2f}"

        ])



    print(
        "\nNORMALIZED MODEL BENCHMARK RESULTS\n"
    )


    print(

        tabulate(

            table,

            headers=[

                "Model",
                "Test",
                "Total Tokens",
                "Latency",
                "Tokens/sec"

            ],

            tablefmt="grid"

        )

    )



    summary = {}


    for r in results:

        model = r["model"]


        if model not in summary:

            summary[model] = {

                "tokens": [],
                "latency": [],
                "speed": []

            }


        summary[model]["tokens"].append(
            r["total_tokens"]
        )

        summary[model]["latency"].append(
            r["latency_seconds"]
        )

        summary[model]["speed"].append(
            r["tokens_per_second"]
        )



    print(
        "\nNORMALIZED MODEL SUMMARY\n"
    )


    summary_table = []


    for model, data in summary.items():

        summary_table.append([

            model,

            round(
                sum(data["tokens"]) /
                len(data["tokens"]),
                2
            ),

            round(
                sum(data["latency"]) /
                len(data["latency"]),
                2
            ),

            round(
                sum(data["speed"]) /
                len(data["speed"]),
                2
            )

        ])



    print(

        tabulate(

            summary_table,

            headers=[

                "Model",
                "Avg Tokens",
                "Avg Latency(s)",
                "Avg Tokens/sec"

            ],

            tablefmt="grid"

        )

    )



if __name__ == "__main__":
    main()
