#!/usr/bin/env python3
"""
Analyze PET metadata from JSON files to identify dataset-specific characteristics
between ADNI and PPMI datasets.
"""

import json
import os
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_pet_metadata(data_root):
    """Analyze all PET JSON files to extract dataset characteristics."""
    
    # Data structures to store metadata
    metadata = []
    scanner_stats = defaultdict(Counter)
    institution_stats = defaultdict(Counter)
    tracer_stats = defaultdict(Counter)
    reconstruction_stats = defaultdict(Counter)
    
    # Find all JSON files
    json_files = list(Path(data_root).rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract source from path
            path_parts = json_file.parts
            if 'ADNI' in path_parts:
                source = 'ADNI'
                label = path_parts[path_parts.index('ADNI') + 1]  # AD or CN
            elif 'PPMI' in path_parts:
                source = 'PPMI'
                label = path_parts[path_parts.index('PPMI') + 1]  # PD or CN
            else:
                continue
            
            # Extract key metadata
            record = {
                'source': source,
                'label': label,
                'subject_id': path_parts[-1].replace('.json', ''),
                'manufacturer': data.get('Manufacturer', 'Unknown'),
                'model': data.get('ManufacturersModelName', 'Unknown'),
                'institution': data.get('InstitutionName', 'Unknown'),
                'tracer': data.get('Radiopharmaceutical', 'Unknown'),
                'reconstruction': data.get('ReconstructionMethod', 'Unknown'),
                'software_version': data.get('SoftwareVersions', 'Unknown'),
                'slice_thickness': data.get('SliceThickness', 'Unknown'),
                'units': data.get('Units', 'Unknown'),
                'decay_correction': data.get('DecayCorrection', 'Unknown'),
                'attenuation_correction': data.get('AttenuationCorrectionMethod', 'Unknown'),
                'frame_duration': data.get('FrameDuration', 'Unknown'),
                'injected_dose': data.get('InjectedRadioactivity', data.get('RadionuclideTotalDose', 'Unknown')),
                'dose_units': data.get('InjectedRadioactivityUnits', 'Unknown')
            }
            
            metadata.append(record)
            
            # Update statistics
            scanner_stats[source][record['manufacturer']] += 1
            institution_stats[source][record['institution']] += 1
            tracer_stats[source][record['tracer']] += 1
            reconstruction_stats[source][record['reconstruction']] += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return metadata, scanner_stats, institution_stats, tracer_stats, reconstruction_stats

def print_analysis_results(metadata, scanner_stats, institution_stats, tracer_stats, reconstruction_stats):
    """Print comprehensive analysis results."""
    
    print("=" * 80)
    print("PET DATASET METADATA ANALYSIS")
    print("=" * 80)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(metadata)
    
    print(f"\nTotal subjects analyzed: {len(df)}")
    print(f"ADNI subjects: {len(df[df['source'] == 'ADNI'])}")
    print(f"PPMI subjects: {len(df[df['source'] == 'PPMI'])}")
    
    print("\n" + "=" * 50)
    print("LABEL DISTRIBUTION BY SOURCE")
    print("=" * 50)
    label_source_dist = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    print(label_source_dist)
    
    print("\n" + "=" * 50)
    print("SCANNER MANUFACTURERS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        for manufacturer, count in scanner_stats[source].most_common():
            print(f"  {manufacturer}: {count}")
    
    print("\n" + "=" * 50)
    print("SCANNER MODELS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        models = df[df['source'] == source]['model'].value_counts()
        for model, count in models.items():
            print(f"  {model}: {count}")
    
    print("\n" + "=" * 50)
    print("INSTITUTIONS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        institutions = df[df['source'] == source]['institution'].value_counts()
        for institution, count in institutions.items():
            print(f"  {institution}: {count}")
    
    print("\n" + "=" * 50)
    print("RADIOPHARMACEUTICALS (TRACERS) BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        tracers = df[df['source'] == source]['tracer'].value_counts()
        for tracer, count in tracers.items():
            print(f"  {tracer}: {count}")
    
    print("\n" + "=" * 50)
    print("RECONSTRUCTION METHODS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        reconstructions = df[df['source'] == source]['reconstruction'].value_counts()
        for recon, count in reconstructions.items():
            print(f"  {recon}: {count}")
    
    print("\n" + "=" * 50)
    print("SOFTWARE VERSIONS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        versions = df[df['source'] == source]['software_version'].value_counts()
        for version, count in versions.items():
            print(f"  {version}: {count}")
    
    print("\n" + "=" * 50)
    print("SLICE THICKNESS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        thicknesses = df[df['source'] == source]['slice_thickness'].value_counts()
        for thickness, count in thicknesses.items():
            print(f"  {thickness}: {count}")
    
    print("\n" + "=" * 50)
    print("ATTENUATION CORRECTION METHODS BY SOURCE")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        print(f"\n{source}:")
        ac_methods = df[df['source'] == source]['attenuation_correction'].value_counts()
        for method, count in ac_methods.items():
            print(f"  {method}: {count}")
    
    print("\n" + "=" * 50)
    print("DETAILED BREAKDOWN BY SOURCE AND LABEL")
    print("=" * 50)
    for source in ['ADNI', 'PPMI']:
        for label in df[df['source'] == source]['label'].unique():
            subset = df[(df['source'] == source) & (df['label'] == label)]
            print(f"\n{source} - {label} (n={len(subset)}):")
            print(f"  Manufacturers: {subset['manufacturer'].value_counts().to_dict()}")
            print(f"  Tracers: {subset['tracer'].value_counts().to_dict()}")
            print(f"  Institutions: {subset['institution'].value_counts().to_dict()}")

def save_detailed_results(metadata, output_file):
    """Save detailed results to CSV for further analysis."""
    df = pd.DataFrame(metadata)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    # Path to raw PET data
    data_root = "/home/jsto890/reseng202500013-ndd-ml/data/raw/PET"
    
    print("Analyzing PET metadata...")
    metadata, scanner_stats, institution_stats, tracer_stats, reconstruction_stats = analyze_pet_metadata(data_root)
    
    print_analysis_results(metadata, scanner_stats, institution_stats, tracer_stats, reconstruction_stats)
    
    # Save detailed results
    save_detailed_results(metadata, "pet_metadata_analysis.csv") 