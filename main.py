import os

import benchmarkplatform
import resultcomparator

dirname = os.path.dirname(__file__)

if __name__ == '__main__':
    # Overall benchmarking across databases/algorithms.
    # Writes results/overall_results.txt with FP, FN, Recall, Precision, Error Rate, F1 and 95% CI (bootstrap on F1).

    databases_to_run = [
        #"MITLT",
        #"Fantasia",
        #"PTT",
        #"MITAR",
        "MITNST",
    ]

    algos_to_run = [
        "pan", 
        "hamilton", 
        "elgendi", 
        "shaik", 
        "park", 
        "arteagaFalconi", 
        "xu", 
        "nguyen", 
        "zhai", 
        "kumari", 
        "xia",
        "zahid", 
        "laitala", 
        "han_cnn", 
        "han_rnn", 
        "xiang", 
        "celik",
    ]

    overall_by_algo = {
        algo: {"tp": 0, "fp": 0, "fn": 0, "tp_list": [], "fp_list": [], "fn_list": []}
        for algo in algos_to_run
    }

    results_dir = os.path.join(dirname, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for db in databases_to_run:
        db_results_path = os.path.join(results_dir, f'{db}_results.txt')
        with open(db_results_path, 'w') as f:
            f.write(f"=========== Database: {db} ===========\n")
            f.write("Metrics: FP, FN, Recall, Precision, Error Rate, F1-Score, 95% CI (F1)\n\n")

        for algo in algos_to_run:
            with open(db_results_path, 'a') as f:
                f.write(f"###### Algorithm: {algo} ######\n")

            metrics = benchmarkplatform.r_peak_benchmarker(
                algo=algo,
                database=db,
                individualresults=True,
            )

            overall_by_algo[algo]["tp"] += metrics["tp"]
            overall_by_algo[algo]["fp"] += metrics["fp"]
            overall_by_algo[algo]["fn"] += metrics["fn"]
            overall_by_algo[algo]["tp_list"].extend(metrics["tp_list"])
            overall_by_algo[algo]["fp_list"].extend(metrics["fp_list"])
            overall_by_algo[algo]["fn_list"].extend(metrics["fn_list"])

    overall_path = os.path.join(results_dir, 'overall_results.txt')
    with open(overall_path, 'w') as f:
        f.write("=========== Overall results across all databases ===========\n")
        f.write(f"Databases: {', '.join(databases_to_run)}\n")
        f.write("Metrics: FP, FN, Recall, Precision, Error Rate, F1-Score, 95% CI (F1)\n\n")

        for algo in algos_to_run:
            tp = overall_by_algo[algo]["tp"]
            fp = overall_by_algo[algo]["fp"]
            fn = overall_by_algo[algo]["fn"]

            precision, recall, _, f1, error_rate = resultcomparator.overallevaluation(tp, fp, fn)

            tp_list = overall_by_algo[algo]["tp_list"]
            fp_list = overall_by_algo[algo]["fp_list"]
            fn_list = overall_by_algo[algo]["fn_list"]
            if len(tp_list) > 0:
                ci_lower, ci_upper = resultcomparator.bootstrap_f1_ci(tp_list, fp_list, fn_list)
            else:
                ci_lower, ci_upper = 0, 0

            f.write(f"###### Algorithm: {algo} ######\n")
            f.write(
                f"FP: {fp}, FN: {fn}, Recall: {recall:.4f}, Precision: {precision:.4f}, "
                f"Error Rate: {error_rate:.4f}, F1-Score: {f1:.4f} (95% CI: [{ci_lower:.4f}-{ci_upper:.4f}])\n\n"
            )

    print(f"Saved overall results to: {overall_path}")