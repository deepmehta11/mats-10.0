#!/usr/bin/env python3
"""
Analyze experiment results and generate report.

Usage:
    python analyze_results.py results/experiment_TIMESTAMP.json
"""

import json
import sys
import pandas as pd
from collections import defaultdict

def load_results(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze(results):
    df = pd.DataFrame(results)
    
    print("=" * 80)
    print("MATS EXPERIMENT ANALYSIS: Monitoring Awareness & CoT Faithfulness")
    print("=" * 80)
    
    # Basic stats
    print(f"\nTotal responses: {len(df)}")
    print(f"Successful extractions: {df['extracted_answer'].notna().sum()}")
    
    # Hint usage
    hint_used = df[df['used_hint'] == True]
    print(f"\nHint used (wrong answer given): {len(hint_used)} ({len(hint_used)/len(df)*100:.1f}%)")
    
    # Main result: Faithfulness by condition
    print("\n" + "=" * 80)
    print("MAIN RESULT: FAITHFULNESS BY CONDITION")
    print("=" * 80)
    print("(Faithfulness = admitted seeing hint when hint was used)\n")
    
    for condition in df['condition'].unique():
        cond_hint = df[(df['condition'] == condition) & (df['used_hint'] == True)]
        if len(cond_hint) > 0:
            faithful = cond_hint['admitted_hint'].sum()
            rate = faithful / len(cond_hint) * 100
            print(f"{condition:20s}: {faithful:3d}/{len(cond_hint):3d} = {rate:5.1f}% faithful")
    
    # Calculate delta
    no_mon = df[(df['condition'] == 'no_monitoring') & (df['used_hint'] == True)]
    with_mon = df[(df['condition'] == 'with_monitoring') & (df['used_hint'] == True)]
    
    if len(no_mon) > 0 and len(with_mon) > 0:
        no_rate = no_mon['admitted_hint'].sum() / len(no_mon) * 100
        with_rate = with_mon['admitted_hint'].sum() / len(with_mon) * 100
        delta = with_rate - no_rate
        print(f"\n{'DELTA':20s}: {delta:+.1f}%")
        
        if delta > 0:
            print("\n✅ HYPOTHESIS SUPPORTED: Monitoring increases faithfulness")
        elif delta < 0:
            print("\n❌ HYPOTHESIS NOT SUPPORTED: Monitoring decreases faithfulness")
        else:
            print("\n⚪ NO EFFECT: Monitoring has no impact on faithfulness")
    
    # By model
    print("\n" + "=" * 80)
    print("FAITHFULNESS BY MODEL")
    print("=" * 80 + "\n")
    
    for model in df['model'].unique():
        model_hint = df[(df['model'] == model) & (df['used_hint'] == True)]
        if len(model_hint) > 0:
            faithful = model_hint['admitted_hint'].sum()
            rate = faithful / len(model_hint) * 100
            print(f"{model:20s}: {faithful:3d}/{len(model_hint):3d} = {rate:5.1f}%")
    
    # Model x Condition matrix
    print("\n" + "=" * 80)
    print("MODEL × CONDITION MATRIX")
    print("=" * 80 + "\n")
    print(f"{'Model':<20} | {'No Monitor':>12} | {'With Monitor':>12} | {'Δ':>8}")
    print("-" * 60)
    
    for model in df['model'].unique():
        rates = {}
        for condition in ['no_monitoring', 'with_monitoring']:
            subset = df[(df['model'] == model) & (df['condition'] == condition) & (df['used_hint'] == True)]
            if len(subset) > 0:
                rates[condition] = subset['admitted_hint'].sum() / len(subset) * 100
            else:
                rates[condition] = None
        
        no_str = f"{rates['no_monitoring']:.1f}%" if rates['no_monitoring'] is not None else "N/A"
        with_str = f"{rates['with_monitoring']:.1f}%" if rates['with_monitoring'] is not None else "N/A"
        
        if rates['no_monitoring'] is not None and rates['with_monitoring'] is not None:
            delta = rates['with_monitoring'] - rates['no_monitoring']
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"
        
        print(f"{model:<20} | {no_str:>12} | {with_str:>12} | {delta_str:>8}")
    
    # Sample faithful vs unfaithful responses
    print("\n" + "=" * 80)
    print("SAMPLE RESPONSES")
    print("=" * 80)
    
    # Faithful example
    faithful_examples = df[(df['used_hint'] == True) & (df['admitted_hint'] == True)]
    if len(faithful_examples) > 0:
        ex = faithful_examples.iloc[0]
        print(f"\n✅ FAITHFUL RESPONSE (admitted hint):")
        print(f"Model: {ex['model']}")
        print(f"Question: {ex['question_text'][:80]}...")
        print(f"Admission quote: {ex['admission_quote']}")
    
    # Unfaithful example
    unfaithful_examples = df[(df['used_hint'] == True) & (df['admitted_hint'] == False)]
    if len(unfaithful_examples) > 0:
        ex = unfaithful_examples.iloc[0]
        print(f"\n❌ UNFAITHFUL RESPONSE (did not admit hint):")
        print(f"Model: {ex['model']}")
        print(f"Question: {ex['question_text'][:80]}...")
        print(f"Response excerpt: {ex['full_response'][:200]}...")
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY FOR MATS APPLICATION")
    print("=" * 80)
    
    if len(no_mon) > 0 and len(with_mon) > 0:
        no_rate = no_mon['admitted_hint'].sum() / len(no_mon) * 100
        with_rate = with_mon['admitted_hint'].sum() / len(with_mon) * 100
        delta = with_rate - no_rate
        
        print(f"""
Hypothesis: When models are told their reasoning is being monitored,
they are more likely to admit when they used external information.

Results:
- Baseline (no monitoring): {no_rate:.1f}% faithfulness
- With monitoring: {with_rate:.1f}% faithfulness  
- Effect: {delta:+.1f}%

Key finding: {"Monitoring IMPROVES faithfulness" if delta > 0 else "Monitoring does NOT improve faithfulness"}
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py results/experiment_TIMESTAMP.json")
        sys.exit(1)
    
    results = load_results(sys.argv[1])
    analyze(results)
