import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import multiprocessing
import pandas as pd
import time

def process_peptide(peptide):
    """计算单个肽序列的描述符，并显示当前进程 ID"""
    process_id = os.getpid()  # 获取当前进程 ID
    if len(peptide) < 5:
        raise Exception("Peptide length is unsuitable.")

    peptides_descriptor = {}
    peptide = str(peptide)
    AAC = AAComposition.CalculateAAComposition(peptide)
    DIP = AAComposition.CalculateDipeptideComposition(peptide)
    MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
    CCTD = CTD.CalculateCTD(peptide)
    QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
    PAAC = PseudoAAC._GetPseudoAAC(peptide, lamda=5)
    APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
    Basic = BasicDes.cal_discriptors(peptide)

    peptides_descriptor.update(AAC)
    peptides_descriptor.update(DIP)
    peptides_descriptor.update(MBA)
    peptides_descriptor.update(CCTD)
    peptides_descriptor.update(QSO)
    peptides_descriptor.update(PAAC)
    peptides_descriptor.update(APAAC)
    peptides_descriptor.update(Basic)

    return (process_id, peptides_descriptor)  # 返回当前进程 ID 和计算结果

# 在模块级别定义函数（而不是在 cal_pep_parallel 内部）
def _process_wrapper(x):
    return (x[0], process_peptide(x[1]))


def cal_pep_parallel(peptides, sequence, results, types, output_path=None, num_workers=4):
    """
    多进程计算肽序列特征，显示进程 ID，并每处理 1000 个打印一次进度
    保持结果顺序与输入顺序一致
    """
    print(f"开始计算，共有 {len(peptides)} 条序列，使用 {num_workers} 个进程")

    # 为每个肽分配一个索引，用于保持顺序
    indexed_peptides = list(enumerate(peptides))
    process_usage = {}  # 统计每个进程处理的任务数
    ordered_results = [None] * len(peptides)  # 预分配列表，用于按原始顺序存储结果

    with multiprocessing.Pool(num_workers) as pool:
        for i, (original_idx, (process_id, descriptor)) in enumerate(
            pool.imap_unordered(
                _process_wrapper, 
                indexed_peptides
            ), 
            1
        ):
            # 将结果存入原始索引位置
            ordered_results[original_idx] = descriptor

            # 统计每个进程的工作量
            if process_id not in process_usage:
                process_usage[process_id] = 0
            process_usage[process_id] += 1

            if i % 1000 == 0:
                print(f"已处理 {i}/{len(peptides)} 条序列 | 进程 ID: {process_id} | 进程统计: {process_usage}")

    print(f"所有序列处理完成！共 {len(ordered_results)} 条")
    print(f"进程工作统计: {process_usage}")

    # 现在ordered_results的顺序与输入peptides的顺序一致
    feature_df = pd.DataFrame(ordered_results)
    output_df = pd.concat([sequence, feature_df, results, types], axis=1)

    if output_path:
        output_df.to_csv(output_path, index=False)
        print(f"结果已保存至: {output_path}")

    return output_df