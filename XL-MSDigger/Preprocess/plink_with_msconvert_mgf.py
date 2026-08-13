import pandas as pd
import numpy as np
import math
import os
import sys
import re
from Preprocess.mass_cal import mz_cal as M


class plink_with_msconvert_mgf():
    """适用于msconvert软件生成的mgf文件和plink2搜库结果的处理类"""

    def __init__(self, crosslinker='DSS', fragment_ppm=0.00002, fragment_num=24, min_mz=0, max_mz=1700,
                 intensity_threshold=0, plink_score_cutoff=1, score_higher_better=False):
        self.crosslinker = crosslinker
        self.frag_ppm = fragment_ppm
        self.frag_num = fragment_num
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.intensity_threshold = intensity_threshold
        self.plink_score_cutoff = plink_score_cutoff
        self.score_higher_better = bool(score_higher_better)

    def _best_score_index(self, scores):
        values = np.asarray(scores, dtype=float)

        if self.score_higher_better:
            return int(np.argmax(values))

        return int(np.argmin(values))

    def extract_from_plink_xl_peptide(self, xl_peptide):
        """用于从plink的交联肽序列中提取两条肽和位点的信息"""
        id1 = xl_peptide.find('-')
        pep1 = xl_peptide[:id1]
        pep2 = xl_peptide[(id1 + 1):]
        id1 = pep1.find('(')
        id2 = pep1.find(')')
        peptide1 = pep1[:id1]
        site1 = pep1[(id1 + 1):id2]
        id1 = pep2.find('(')
        id2 = pep2.find(')')
        peptide2 = pep2[:id1]
        site2 = pep2[(id1 + 1):id2]
        return peptide1, peptide2, int(site1), int(site2)

    def modif_xlpeptide(self, pep1, pep2, mod):
        len1 = len(pep1)
        len2 = len(pep2)
        if 'M' in mod:
            find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]                         
            id_list = find_all(mod, 'M')
            for id in id_list:
                if mod[(id + 4)] == ')':
                    modid = int(mod[(id + 3)])
                    if modid < (len1 + len2 + 1):
                        if modid > len1:
                            b = list(pep2)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep2 = ''.join(b)
                        else:
                            b = list(pep1)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep1 = ''.join(b)
                elif mod[(id + 5)] == ')':
                    modid = int(mod[(id + 3):(id + 5)])
                    if modid < (len1 + len2 + 1):
                        if modid > len1:
                            b = list(pep2)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep2 = ''.join(b)
                        else:
                            b = list(pep1)
                            if b[(modid - len1 - 1)] == 'M':
                                b[(modid - len1 - 1)] = 'e'
                            pep1 = ''.join(b)
        return pep1, pep2

    def extract_from_combine_peptide(self, combine_peptide):
        """从combine_peptide中提取两条肽和位点的信息"""
        id1 = combine_peptide.find('X')
        pep1 = combine_peptide[:id1]
        pep2 = combine_peptide[(id1 + 1):]
        site1 = pep1.find('U')
        peptide1 = pep1.replace('U', '')
        site2 = pep2.find('U')
        peptide2 = pep2.replace('U', '')
        return peptide1, peptide2, site1, site2

    def calculate_ccs(self, peptide_m_z, peptide_charge, peptide_k0):
        """计算CCS值"""
        m = 28.00615
        t = 304.7527
        coeff = 18500 * peptide_charge * math.sqrt(
            (peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
        ccs = coeff * peptide_k0
        return ccs

    def build_mgf_metadata_index(self, mgf_dir):
        """
        Parse MGF metadata once and build:

            TITLE -> (RT, ion_mobility)

        TITLE-embedded metadata:
            $<RT>$
            #<ion mobility>#

        takes precedence over explicit metadata fields:

            RTINSECONDS=
            ION_MOBILITY=

        Returns
        -------
        metadata_index : dict
            Mapping from spectrum TITLE to (rt, ion_mobility).

        has_explicit_ion_mobility : bool
            True only when explicit ION_MOBILITY= metadata
            occurs in the MGF. This preserves the existing
            decision about whether to use the CCS workflow.
        """

        metadata_index = {}

        number_pattern = (
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?"
        )

        rt_pattern = re.compile(
            rf"\$({number_pattern})\$"
        )

        mobility_pattern = re.compile(
            rf"#({number_pattern})#"
        )

        numeric_pattern = re.compile(
            number_pattern
        )

        current_title = None
        current_rt = None
        current_ion_mobility = None

        has_explicit_ion_mobility = False


        def store_current():
            if current_title is not None:
                metadata_index[current_title] = (
                    current_rt,
                    current_ion_mobility
                )


        with open(
            mgf_dir,
            "r",
            errors="replace"
        ) as handle:

            for raw_line in handle:

                line = raw_line.strip()

                if line == "BEGIN IONS":
                    current_title = None
                    current_rt = None
                    current_ion_mobility = None
                    continue


                if line.startswith("TITLE="):

                    current_title = (
                        line[len("TITLE="):]
                        .strip()
                    )

                    rt_match = rt_pattern.search(
                        current_title
                    )

                    mobility_match = (
                        mobility_pattern.search(
                            current_title
                        )
                    )

                    if rt_match is not None:
                        current_rt = float(
                            rt_match.group(1)
                        )

                    if mobility_match is not None:
                        current_ion_mobility = float(
                            mobility_match.group(1)
                        )

                    continue


                if (
                    line.startswith("RTINSECONDS=")
                    and current_rt is None
                ):

                    try:
                        # Explicit MGF RTINSECONDS is in seconds.
                        # Deep4D-XL RT models operate on minute-scale
                        # chromatographic retention times.
                        current_rt = (
                            float(
                                line.split(
                                    "=",
                                    1
                                )[1].strip()
                            )
                            / 60.0
                        )
                    except (ValueError, TypeError):
                        pass

                    continue


                if line.startswith("ION_MOBILITY="):

                    has_explicit_ion_mobility = True

                    if current_ion_mobility is None:

                        payload = line.split(
                            "=",
                            1
                        )[1].strip()

                        numeric_values = (
                            numeric_pattern.findall(
                                payload
                            )
                        )

                        if numeric_values:
                            try:
                                current_ion_mobility = float(
                                    numeric_values[-1]
                                )
                            except (
                                ValueError,
                                TypeError
                            ):
                                pass

                    continue


                if line == "END IONS":

                    store_current()

                    current_title = None
                    current_rt = None
                    current_ion_mobility = None


        # Defensive handling for malformed files lacking
        # a final END IONS.
        store_current()

        return (
            metadata_index,
            has_explicit_ion_mobility
        )


    def parse_mgf_info(self, spectrum, title):
        """
        Parse retention time and ion-mobility information from an MGF
        spectrum.

        Supports both:
          1. Metadata embedded in TITLE:
                 $<RT>$
                 #<ion mobility>#
          2. Standard MGF metadata lines:
                 RTINSECONDS=
                 ION_MOBILITY=

        TITLE-embedded values take precedence when present.
        """
        rt = None
        ion_mobility = None

        title = str(title).strip()

        # Fast path: a pre-built TITLE -> (RT, k0)
        # metadata index.
        if isinstance(spectrum, dict):
            return spectrum.get(
                title,
                (None, None)
            )

        number_pattern = (
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
            r"(?:[eE][-+]?\d+)?"
        )

        # Metadata embedded directly in TITLE
        rt_match = re.search(
            rf"\$({number_pattern})\$",
            title
        )

        mobility_match = re.search(
            rf"#({number_pattern})#",
            title
        )

        if rt_match is not None:
            rt = float(rt_match.group(1))

        if mobility_match is not None:
            ion_mobility = float(
                mobility_match.group(1)
            )

        # Fall back to explicit MGF metadata fields
        if spectrum is None:
            return rt, ion_mobility

        spectrum_text = np.asarray(
            [
                str(line).strip()
                for line in np.asarray(spectrum).reshape(-1)
            ],
            dtype=object
        )

        title_line = f"TITLE={title}"

        title_indices = np.flatnonzero(
            spectrum_text == title_line
        )

        if len(title_indices) == 0:
            return rt, ion_mobility

        start_index = int(title_indices[0]) + 1

        for i in range(
            start_index,
            min(start_index + 10, len(spectrum_text))
        ):
            line = spectrum_text[i]

            if (
                rt is None
                and line.startswith("RTINSECONDS=")
            ):
                try:
                    # Explicit MGF RTINSECONDS is in seconds.
                    # Convert to minutes for Deep4D-XL.
                    rt = (
                        float(
                            line.split("=", 1)[1].strip()
                        )
                        / 60.0
                    )
                except (ValueError, TypeError):
                    pass

            elif (
                ion_mobility is None
                and line.startswith("ION_MOBILITY=")
            ):
                payload = line.split(
                    "=",
                    1
                )[1].strip()

                numeric_values = re.findall(
                    number_pattern,
                    payload
                )

                if numeric_values:
                    try:
                        ion_mobility = float(
                            numeric_values[-1]
                        )
                    except (ValueError, TypeError):
                        pass

            elif (
                line.startswith("CHARGE=")
                or line == "END IONS"
            ):
                break

        return rt, ion_mobility


    def change_plink_filter_crosslink(self, plinkfile_dir, spectrum=None):
        """处理plink3生成的filtered_crosslinked_spectra文件"""
                                  
        plinkfile_dir = plinkfile_dir + '/reports/'
        spectra_file = None

        for filename in os.listdir(plinkfile_dir):
            if filename.startswith('._'):
                continue
            if 'filtered_crosslinked_spectra.csv' in filename or 'filtered_cross-linked_spectra.csv' in filename:
                spectra_file = plinkfile_dir + filename
                break

        if spectra_file is None:
            raise FileNotFoundError("未找到filtered_crosslinked_spectra.csv文件")

              
        data = pd.read_csv(spectra_file, header=0, index_col=False, engine='python', on_bad_lines='warn')
                                              
        if 'Charge' in data.columns:
            first_charge = data.iloc[0]['Charge']
            try:
                int(first_charge)
            except (ValueError, TypeError):
                print("Detected column misalignment, fixing...")

                                 
                original_columns = list(data.columns)[:-1]

                             
                data['Title'] = data['Title'].astype(str) + ',' + data['Charge'].astype(str)

                              
                data = data.drop('Charge', axis=1)

                              
                data.columns = original_columns

               
        cl_peptide_list, m_z_list, charge_list, peptide1_list, site1_list, peptide2_list, site2_list = [], [], [], [], [], [], []
        rt_list, k0_list, ccs_list, score_list, precursor_Mass_Error_list, intensity_list, type_list = [], [], [], [], [], [], []
        protein_list, protein_type_list, title_list, filename_list, cmpd_list = [], [], [], [], []

        for index, row in data.iterrows():
            peptide = row['Peptide']
            peptide1, peptide2, site1, site2 = self.extract_from_plink_xl_peptide(peptide)

                  
            modif = str(row['Modifications']) if pd.notna(row['Modifications']) else ''
            peptide1, peptide2 = self.modif_xlpeptide(peptide1, peptide2, modif)

                    
            if any(char in peptide for char in ['U', 'O', 'X', 'B', 'J', 'Z']):
                continue

                                
            title = row['Title']
            charge = row['Charge']
                                                        
            if 'Precursor_Mass' in row:
                precursor_mass = row['Precursor_Mass']
            else:
                precursor_mass = row['Precursor_MH']

            score = row['Score']                         
            precursor_mass_error = row['Precursor_Mass_Error(ppm)']
            protein = row['Proteins']
            protein_type = row['Protein_Type']

                   
            m_z = (float(precursor_mass) + (int(charge) - 1) * 1.00728) / int(charge)

                                      
            rt = None
            k0 = None
            ccs = None
            intensity = 1.0

            if spectrum is not None:
                rt, k0 = self.parse_mgf_info(spectrum, title)
                if k0 is not None:
                    ccs = self.calculate_ccs(m_z, int(charge), float(k0))

                          
            if '.' in title:
                filename = title.split('.')[0]
                cmpd = ''
            else:
                filename = title
                cmpd = ''

                   
            cl_peptide_list.append(peptide)
            m_z_list.append(m_z)
            charge_list.append(charge)
            peptide1_list.append(peptide1)
            peptide2_list.append(peptide2)
            site1_list.append(site1)
            site2_list.append(site2)
            rt_list.append(rt)
            k0_list.append(k0)
            ccs_list.append(ccs)
            score_list.append(score)
            precursor_Mass_Error_list.append(precursor_mass_error)
            title_list.append(title)
            filename_list.append(filename)
            intensity_list.append(intensity)
            cmpd_list.append(cmpd)
            type_list.append(3)
            protein_list.append(protein)
            protein_type_list.append(protein_type)

                                       
        result_data = pd.DataFrame()
        result_data['title'] = title_list
        result_data['filename'] = filename_list
        result_data['peptide'] = cl_peptide_list
        result_data['m_z'] = m_z_list
        result_data['charge'] = charge_list
        result_data['peptide1'] = peptide1_list
        result_data['peptide2'] = peptide2_list
        result_data['site1'] = site1_list
        result_data['site2'] = site2_list
        result_data['rt'] = rt_list
        result_data['k0'] = k0_list
        result_data['ccs'] = ccs_list
        result_data['score'] = score_list
        result_data['precursor_Mass_Error(ppm)'] = precursor_Mass_Error_list
        result_data['intensity'] = intensity_list
        result_data['cmpd'] = cmpd_list
        result_data['peptide_type'] = type_list
        result_data['protein'] = protein_list
        result_data['protein_type'] = protein_type_list

                             
        combine_pep, combine_pep_z = [], []
        for i in range(len(peptide1_list)):
            pep1 = peptide1_list[i]
            pep2 = peptide2_list[i]
            s1 = int(site1_list[i])
            s2 = int(site2_list[i])
            z = int(charge_list[i])

            pep1_list = list(pep1)
            pep2_list = list(pep2)
            pep1_list.insert(s1, 'U')
            pep2_list.insert(s2, 'U')
            pep1_mod = ''.join(pep1_list)
            pep2_mod = ''.join(pep2_list)

            pep = pep1_mod + 'X' + pep2_mod
            pep_z = pep1_mod + 'X' + pep2_mod + str(z)
            combine_pep.append(pep)
            combine_pep_z.append(pep_z)

        result_data['combine_peptide'] = combine_pep
        result_data['combine_peptide_z'] = combine_pep_z

        return result_data

    def change_plink_total_results(self, plinkfile_dir, spectrum=None):
        """处理plink的总结果文件 - 使用总鉴定文件（最短文件名）"""
        plinkfile_dir = plinkfile_dir + '/reports/'
        listdir = os.listdir(plinkfile_dir)
                           
        file = plinkfile_dir + min(listdir, key=len)

        data = pd.read_csv(file)
        data = data[data['Peptide_Type'] == 3]          

               
        m_z_list, charge_list, peptide1_list, site1_list, peptide2_list, site2_list = [], [], [], [], [], []
        rt_list, k0_list, ccs_list, combine_pep, combine_pep_z, len1_list, len2_list = [], [], [], [], [], [], []

        for index, row in data.iterrows():
            peptide = row['Peptide']
            modif = str(row['Modifications']) if pd.notna(row['Modifications']) else ''
            charge = row['Charge']

            peptide1, peptide2, site1, site2 = self.extract_from_plink_xl_peptide(peptide)
            len1 = len(peptide1)
            len2 = len(peptide2)
            peptide1, peptide2 = self.modif_xlpeptide(peptide1, peptide2, modif)

            z = int(charge)
            pep1_list = list(peptide1)
            pep2_list = list(peptide2)
            pep1_list.insert(site1, 'U')
            pep2_list.insert(site2, 'U')
            pep1_mod = ''.join(pep1_list)
            pep2_mod = ''.join(pep2_list)

            pep = pep1_mod + 'X' + pep2_mod
            pep_z = pep1_mod + 'X' + pep2_mod + str(z)

                                                   
            mass = row['Precursor_MH']
            title = row['Title']

                                      
            rt = None
            k0 = None
            ccs = None
            if spectrum is not None:
                rt, k0 = self.parse_mgf_info(spectrum, title)

            m_z = (float(mass) + (int(charge) - 1) * 1.00728) / int(charge)
            if k0 is not None:
                ccs = self.calculate_ccs(m_z, int(charge), float(k0))

            m_z_list.append(m_z)
            charge_list.append(charge)
            peptide1_list.append(peptide1)
            peptide2_list.append(peptide2)
            len1_list.append(len1)
            len2_list.append(len2)
            site1_list.append(site1)
            site2_list.append(site2)
            rt_list.append(rt)
            k0_list.append(k0)
            ccs_list.append(ccs)
            combine_pep.append(pep)
            combine_pep_z.append(pep_z)

                     
        data['m_z'] = m_z_list
        data['charge'] = charge_list
        data['peptide1'] = peptide1_list
        data['peptide2'] = peptide2_list
        data['len1'] = len1_list
        data['len2'] = len2_list
        data['site1'] = site1_list
        data['site2'] = site2_list
        data['rt'] = rt_list
        data['k0'] = k0_list
        data['ccs'] = ccs_list
        data['combine_peptide'] = combine_pep
        data['combine_peptide_z'] = combine_pep_z

              
        data = data[data['Charge'] < 6]
        data = data[data['len1'] < 50]
        data = data[data['len2'] < 50]
        
                                   
        noncanonical_aas = set(["B", "J", "O", "U", "X", "Z"])
        def contains_noncanonical(peptide: str):
            return any(aa in noncanonical_aas for aa in peptide)

        data = data[~data['peptide1'].apply(contains_noncanonical)]
        data = data[~data['peptide2'].apply(contains_noncanonical)]
                    
        data.rename(columns={'Title': 'title', 'Score': 'score'}, inplace=True)

        return data

    def build_mgf_index(self, mgf_dir):
        """
        Parse the MGF file once and build a dictionary mapping each
        TITLE to its experimental m/z and intensity arrays.
        """
        mgf_index = {}

        current_title = None
        current_mz = []
        current_intensity = []

        with open(mgf_dir, "r", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()

                if line == "BEGIN IONS":
                    current_title = None
                    current_mz = []
                    current_intensity = []
                    continue

                if line.startswith("TITLE="):
                    current_title = line[len("TITLE="):].strip()
                    continue

                if line == "END IONS":
                    if current_title is not None:
                        mgf_index[current_title] = (
                            np.asarray(current_mz, dtype=float),
                            np.asarray(current_intensity, dtype=float)
                        )

                    current_title = None
                    current_mz = []
                    current_intensity = []
                    continue

                if current_title is None:
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                try:
                    peak_mz = float(parts[0])
                    peak_intensity = float(parts[1])
                except (ValueError, TypeError):
                    continue

                if (
                    np.isfinite(peak_mz)
                    and np.isfinite(peak_intensity)
                    and peak_mz > 0
                ):
                    current_mz.append(peak_mz)
                    current_intensity.append(peak_intensity)

        return mgf_index


    def match_msms_indexed(self, mgf_index, m_z, title):
        """
        Match theoretical fragment m/z values using a pre-built
        TITLE -> (experimental_mz, experimental_intensity) MGF index.
        """

        title = str(title).strip()

        if title not in mgf_index:
            print(f"WARNING: MGF title not found: {title}")

            matched_mz = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            matched_intensity = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            return matched_mz, matched_intensity

        experimental_mz, experimental_intensity = (
            mgf_index[title]
        )

        if len(experimental_mz) == 0:
            print(f"WARNING: No peaks parsed for: {title}")

            matched_mz = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            matched_intensity = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            return matched_mz, matched_intensity

        matched_mz = []
        matched_intensity = []

        for theoretical_mz in m_z:

            theoretical_mz = float(theoretical_mz)

            if theoretical_mz == -1:
                matched_mz.append(-1)
                matched_intensity.append(-1)
                continue

            if (
                not np.isfinite(theoretical_mz)
                or theoretical_mz <= 0
            ):
                matched_mz.append(0)
                matched_intensity.append(0)
                continue

            relative_errors = (
                np.abs(
                    experimental_mz - theoretical_mz
                )
                / theoretical_mz
            )

            nearest_index = int(
                np.argmin(relative_errors)
            )

            nearest_error = (
                relative_errors[nearest_index]
            )

            if nearest_error <= self.frag_ppm:

                matched_mz.append(
                    float(
                        experimental_mz[
                            nearest_index
                        ]
                    )
                )

                matched_intensity.append(
                    float(
                        experimental_intensity[
                            nearest_index
                        ]
                    )
                )

            else:
                matched_mz.append(0)
                matched_intensity.append(0)

        return matched_mz, matched_intensity


    def match_msms(self, spectrum, m_z, title):
        """
        Match theoretical fragment m/z values to one MGF spectrum.
        """

        spectrum_text = np.asarray(
            [
                str(line).strip()
                for line in np.asarray(spectrum).reshape(-1)
            ],
            dtype=object
        )

        title = str(title).strip()
        title_line = f"TITLE={title}"

        title_indices = np.flatnonzero(
            spectrum_text == title_line
        )

        if len(title_indices) == 0:
            print(f"WARNING: MGF title not found: {title}")

            matched_mz = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            matched_intensity = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            return matched_mz, matched_intensity

        start_index = int(title_indices[0]) + 1

        experimental_mz = []
        experimental_intensity = []

        for i in range(start_index, len(spectrum_text)):
            line = spectrum_text[i]

            if line == "END IONS":
                break

            if line == "BEGIN IONS" or line.startswith("TITLE="):
                break

            parts = line.split()

            if len(parts) < 2:
                continue

            try:
                peak_mz = float(parts[0])
                peak_intensity = float(parts[1])
            except (ValueError, TypeError):
                continue

            if (
                np.isfinite(peak_mz)
                and np.isfinite(peak_intensity)
                and peak_mz > 0
            ):
                experimental_mz.append(peak_mz)
                experimental_intensity.append(peak_intensity)

        if len(experimental_mz) == 0:
            print(f"WARNING: No peaks parsed for: {title}")

            matched_mz = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            matched_intensity = [
                -1 if float(value) == -1 else 0
                for value in m_z
            ]

            return matched_mz, matched_intensity

        experimental_mz = np.asarray(
            experimental_mz,
            dtype=float
        )

        experimental_intensity = np.asarray(
            experimental_intensity,
            dtype=float
        )

        matched_mz = []
        matched_intensity = []

        for theoretical_mz in m_z:
            theoretical_mz = float(theoretical_mz)

            if theoretical_mz == -1:
                matched_mz.append(-1)
                matched_intensity.append(-1)
                continue

            if (
                not np.isfinite(theoretical_mz)
                or theoretical_mz <= 0
            ):
                matched_mz.append(0)
                matched_intensity.append(0)
                continue

            relative_errors = (
                np.abs(experimental_mz - theoretical_mz)
                / theoretical_mz
            )

            nearest_index = int(np.argmin(relative_errors))
            nearest_error = relative_errors[nearest_index]

            if nearest_error <= self.frag_ppm:
                matched_mz.append(
                    float(experimental_mz[nearest_index])
                )

                matched_intensity.append(
                    float(experimental_intensity[nearest_index])
                )
            else:
                matched_mz.append(0)
                matched_intensity.append(0)

        return matched_mz, matched_intensity

    def filter_plink_precursor_results(self, data):
        """筛选相同肽中score最小的precursor"""
        data1 = data.sort_values('combine_peptide_z', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide_z'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide_z']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[self._best_score_index(q_list)]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide_z'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[self._best_score_index(q_list)]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide_z'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = data6['score'].astype(float)
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

    def filter_plink_peptide_results(self, data):
        """筛选相同肽中score最小的peptide"""
        data1 = data.sort_values('combine_peptide', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[self._best_score_index(q_list)]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[self._best_score_index(q_list)]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
        data6['score'] = data6['score'].astype(float)
        data7 = data6[data6['score'] < float(self.plink_score_cutoff)]
        return data7

    def crosslink_ion_generation(self, peptide1, peptide2):
        """生成交联肽的碎片离子"""
        len1 = len(peptide1)
        len2 = len(peptide2)
        z = ['1', '2', '3', '4', '5']            
        l = ['noloss'] * 5          
        by_1 = (['1b'] * 5 + ['1y'] * 5) * (len1 - 1)
        c = []         
        for i in range(len1 - 1):
            j = i + 1
            c = c + [j] * 10
        for i in range(len2 - 1):
            j = i + 1
            c = c + [j] * 10
        by_2 = (['2b'] * 5 + ['2y'] * 5) * (len2 - 1)
        by = by_1 + by_2
        z = z * 2 * (len1 + len2 - 2)
        l = l * 2 * (len1 + len2 - 2)
        c = np.array(c)         
        by = np.array(by)           
        z = np.array(z)          
        l = np.array(l)         
        data = np.column_stack((c, by, z, l))
        return data

    def choose_top_n(self, data, n):
        """选择前n强度的碎片"""
        name = list(data)
        inten = np.array(data['Fragment_intensity'])
        data1 = np.array(data)
        if len(data1) > n:
            data2 = data1[np.argsort(-inten)]
            data3 = data2[:n, :]
            data3 = pd.DataFrame(data3, columns=name)
            return data3
        else:
            return data

    def genenrate_all_crosslink_fragment(self, plink_data, mgf_dir):
        """生成所有交联碎片离子库"""
        sys.stdout.write("Loading file......\r")

                       
        # Build the MGF spectrum index once for fast candidate lookup.
        mgf_index = self.build_mgf_index(mgf_dir)

                                                           
        data = plink_data
        name = list(data)
        pep1 = np.array(data['peptide1'])
        pep2 = np.array(data['peptide2'])

        num = 0
        for i in range(len(pep1)):
            num = num + (len(pep1[i]) + len(pep2[i]) - 2) * 10

        data_array = np.array(data)
        a = np.array(list(data_array[0, :]) + [0, 0, 0, 0, 0, 0, 0], dtype=object)
        data1 = np.tile(a, [num, 1])
        num = 0
        name = name + ['Fragment_num', 'Fragment_type', 'Fragment_charge', 'Neutral_loss']

        sys.stdout.write("Generating library......\r")

        for i in range(len(pep1)):
            peptide1 = pep1[i]
            peptide2 = pep2[i]
            len1 = (len(pep1[i]) + len(pep2[i]) - 2) * 10
            b = data_array[i, :]
            data_y = np.tile(b, [len1, 1])
            data_x = self.crosslink_ion_generation(peptide1, peptide2)
            data_xy = np.column_stack((data_y, data_x))
            data_xy = pd.DataFrame(data_xy, columns=name)

            xl_peptide = data_xy['combine_peptide']
            by_type = data_xy['Fragment_type']
            Fragment_num = data_xy['Fragment_num']
            Fragment_charge = data_xy['Fragment_charge']
            Neutral_loss = data_xy['Neutral_loss']
            title = data_xy['title']

            m_z = []
            m = M()
            for j in range(len(xl_peptide)):
                mz = m.crosslink_peptide_msms_m_z(xl_peptide.iloc[j], self.crosslinker, by_type.iloc[j],
                                                  int(Fragment_num.iloc[j]), int(Fragment_charge.iloc[j]),
                                                  Neutral_loss.iloc[j])
                mz = round(mz, 5)
                m_z.append(mz)

                        
            m_z_1, inten = self.match_msms_indexed(
                mgf_index,
                m_z,
                title.iloc[0]
            )

            if np.max(inten) > 0:
                inten = np.array(inten) / np.max(inten)
            else:
                inten = np.array(inten)

            m_z = np.array(m_z)
            m_z_1 = np.array(m_z_1)
            data_xy = np.array(data_xy)
            data_xy = np.column_stack((data_xy, m_z, m_z_1, inten))

            charge = int(data_xy[0, 4])
            for k in range(len(data_xy)):
                if int(data_xy[k, -5]) > charge:
                    data_xy[k, -1], data_xy[k, -2], data_xy[k, -3] = -1, -1, -1

            data1[num:(num + len1), :] = data_xy
            num = num + len1

        name = name + ['Fragment_m_z_calculation', 'Fragment_m_z_experiment', 'Fragment_intensity']

        for l in range(len(data1)):
            if data1[l, -1] < 0:
                data1[l, -1] = -1

        data1 = pd.DataFrame(data1, columns=name)
        data1 = data1[data1['Fragment_m_z_experiment'] > self.min_mz]
        data1 = data1[data1['Fragment_m_z_experiment'] < self.max_mz]
        data1 = data1[data1['Fragment_intensity'] > self.intensity_threshold]

        return data1

    def transfer_to_DIANN_format(self, data):
        """转换为DIA-NN格式 - 增加Ion Mobility支持"""
        data2 = pd.DataFrame()
        data2['ModifiedPeptide'] = data['combine_peptide']
        data2['PrecursorCharge'] = data['charge']
        data2['PrecursorMz'] = data['m_z']
        data2['FragmentCharge'] = data['Fragment_charge']
        data2['ProductMz'] = data['Fragment_m_z_calculation']
        data2['Tr_recalibrated'] = data['rt']
        if 'k0' in data.columns and data['k0'].notna().any():
            data2['IonMobility'] = data['k0']
        data2['LibraryIntensity'] = data['Fragment_intensity']
        return data2

    def filter_peptide_with_score_in_msms(self, data):
        """根据电荷态和PSM的score来filter"""
        data1 = data.sort_values('combine_peptide_z', ignore_index=True)             
        peptide_list = np.array(data1['combine_peptide_z'])
        name = list(data1)
        data5 = np.array(data1)                                
        peptide = peptide_list[0]
        index_list = []
        peptide_num = len(set(data1['combine_peptide_z']))
        num = 0
        lenth1 = 0
        for i in range(len(peptide_list)):
            if peptide_list[i] == peptide:
                index_list.append(i)
            else:
                num = num + 1
                data2 = data1.iloc[index_list]
                q_list = list(data2['score'])
                psm_list = list(data2['title'])
                psm = psm_list[self._best_score_index(q_list)]
                data3 = data2[data2['title'] == psm]
                lenth2 = len(data3['combine_peptide_z'])
                data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)                        
                lenth1 = lenth1 + lenth2
                index_list = []
                index_list.append(i)
                peptide = peptide_list[i]
        data2 = data1.iloc[index_list]
        q_list = list(data2['score'])
        psm_list = list(data2['title'])
        psm = psm_list[self._best_score_index(q_list)]
        data3 = data2[data2['title'] == psm]
        lenth2 = len(data3['combine_peptide_z'])
        data5[lenth1:(lenth1 + lenth2), :] = np.array(data3)
        lenth1 = lenth1 + lenth2
        data6 = pd.DataFrame(data5[:lenth1, :], columns=name)
                                   
        data1 = data6
        name = list(data1)
        data3 = np.array(data1)
        peptide = set(list(data1['combine_peptide_z']))
        data2 = np.tile(data3[0, :], (len(peptide) * self.frag_num, 1))
        num1 = 0
        x = 0
        for pep in peptide:
            x = x + 1
            data_x = data1[data1['combine_peptide_z'] == pep]
            data_y = np.array(self.choose_top_n(data_x, self.frag_num))            
            lenth = len(data_y)
            data2[num1:(num1 + lenth), :] = data_y
            num1 = num1 + lenth
        data4 = data2[:num1, :]
        data4 = pd.DataFrame(data4, columns=name)
        data4_1 = data4[
            ['title', 'score', 'peptide1', 'peptide2', 'site1', 'site2', 'combine_peptide', 'combine_peptide_z',
             'charge',
             'm_z', 'rt', 'k0', 'Fragment_charge', 'Fragment_type',
             'Fragment_num', 'Neutral_loss', 'Fragment_intensity', 'Fragment_m_z_calculation']]
        data5 = self.transfer_to_DIANN_format(data4)
        return data6, data4_1, data5

    def process(self, plinkfile, mgf_dir):
        """
        Preprocess pLink cross-link search results and the corresponding
        MGF file.

        Outputs:
          - fine-tuning MS/MS library
          - optional CCS fine-tuning table
          - RT fine-tuning table
          - all-candidate MS/MS fragment table
          - all-candidate precursor/RT/CCS table
          - ion-mobility availability flag
        """
        print("Processing data.......")

        # --------------------------------------------------
        # Load MGF text once for metadata parsing.
        #
        # Keep the repository's explicit ION_MOBILITY=
        # detection here. TITLE-embedded #k0# values are
        # parsed when needed but do not automatically switch
        # the DDA workflow to the CCS model.
        # --------------------------------------------------

        print(
            "Building MGF metadata index..."
        )

        (
            mgf_metadata_index,
            has_ion_mobility
        ) = self.build_mgf_metadata_index(
            mgf_dir
        )

        print(
            "MGF spectra indexed:",
            len(mgf_metadata_index)
        )

        print(
            f"Ion Mobility detected: "
            f"{has_ion_mobility}"
        )

        # --------------------------------------------------
        # Fine-tuning identifications from filtered pLink
        # results
        # --------------------------------------------------

        print(
            "\nGenerating fine-tuning input tables..."
        )

        plink_crosslink_data_all = (
            self.change_plink_filter_crosslink(
                plinkfile,
                mgf_metadata_index
            )
        )

        # --------------------------------------------------
        # Deep4D-XL fine-tuning must use intra-protein
        # cross-linked PSMs only.
        #
        # Do NOT apply this filter to the all-candidate
        # rescoring branch below.
        # --------------------------------------------------

        if "protein_type" in plink_crosslink_data_all.columns:
            protein_type_col = "protein_type"
        elif "Protein_Type" in plink_crosslink_data_all.columns:
            protein_type_col = "Protein_Type"
        else:
            raise RuntimeError(
                "Cannot enforce intra-protein fine-tuning: "
                "protein type column is missing."
            )

        protein_type_values = (
            plink_crosslink_data_all[protein_type_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        intra_mask = protein_type_values.isin(
            [
                "intra-protein",
                "intra protein",
                "intra"
            ]
        )

        plink_crosslink_data = (
            plink_crosslink_data_all.loc[
                intra_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        print(
            "Filtered pLink cross-link PSMs:",
            len(plink_crosslink_data_all)
        )

        print(
            "Intra-protein PSMs used for fine-tuning:",
            len(plink_crosslink_data)
        )

        print(
            "Non-intra PSMs excluded from fine-tuning:",
            len(plink_crosslink_data_all)
            - len(plink_crosslink_data)
        )

        if len(plink_crosslink_data) == 0:
            raise RuntimeError(
                "No intra-protein cross-linked PSMs were found "
                "for Deep4D-XL fine-tuning."
            )

        if has_ion_mobility:
            plink_crosslink_data_ccs = (
                self.filter_plink_precursor_results(
                    plink_crosslink_data
                )
            )

        plink_crosslink_data_rt = (
            self.filter_plink_peptide_results(
                plink_crosslink_data
            )
        )

        complete_normal_library = (
            self.genenrate_all_crosslink_fragment(
                plink_crosslink_data,
                mgf_dir
            )
        )

        # --------------------------------------------------
        # Generate all pLink candidate precursors.
        #
        # This stage was missing from the original process()
        # even though candidate output paths were returned.
        # --------------------------------------------------

        print(
            "\nGenerating all rescoring candidates..."
        )

        all_candidate_data = (
            self.change_plink_total_results(
                plinkfile,
                mgf_metadata_index
            )
        )

        print(
            "Candidate rows:",
            len(all_candidate_data)
        )

        print(
            "Candidate unique titles:",
            all_candidate_data["title"].nunique()
        )

        print(
            "Candidate missing RT:",
            all_candidate_data["rt"].isna().sum()
        )

        all_candidate_msms = (
            self.genenrate_all_crosslink_fragment(
                all_candidate_data,
                mgf_dir
            )
        )

        # --------------------------------------------------
        # Output paths
        # --------------------------------------------------

        if mgf_dir.lower().endswith(".mgf"):
            outputdir = mgf_dir[:-4]
        else:
            outputdir = mgf_dir

        rt_dir = f"{outputdir}_rt.csv"
        ccs_dir = f"{outputdir}_ccs.csv"
        msms_dir = (
            f"{outputdir}_complete_normal_library.csv"
        )

        candidate_msms_dir = (
            f"{outputdir}_all_candidate_msms.csv"
        )

        candidate_rtccs_dir = (
            f"{outputdir}_all_candidate.csv"
        )

        # --------------------------------------------------
        # Write fine-tuning files
        # --------------------------------------------------

        if has_ion_mobility:
            plink_crosslink_data_ccs.to_csv(
                ccs_dir,
                index=False
            )

        plink_crosslink_data_rt.to_csv(
            rt_dir,
            index=False
        )

        complete_normal_library.to_csv(
            msms_dir,
            index=False
        )

        # --------------------------------------------------
        # Write candidate files
        # --------------------------------------------------

        all_candidate_data.to_csv(
            candidate_rtccs_dir,
            index=False
        )

        all_candidate_msms.to_csv(
            candidate_msms_dir,
            index=False
        )

        # --------------------------------------------------
        # Validate that the required outputs now exist
        # --------------------------------------------------

        required_outputs = [
            msms_dir,
            rt_dir,
            candidate_msms_dir,
            candidate_rtccs_dir
        ]

        if has_ion_mobility:
            required_outputs.append(ccs_dir)

        missing_outputs = [
            file
            for file in required_outputs
            if not os.path.exists(file)
        ]

        if missing_outputs:
            raise RuntimeError(
                "Preprocessing failed to create required "
                f"outputs: {missing_outputs}"
            )

        print("\nProcessing completed.")

        print(
            "Fine-tuning MS/MS:",
            msms_dir
        )

        print(
            "Fine-tuning RT:",
            rt_dir
        )

        if has_ion_mobility:
            print(
                "Fine-tuning CCS:",
                ccs_dir
            )

        print(
            "Candidate table:",
            candidate_rtccs_dir
        )

        print(
            "Candidate MS/MS:",
            candidate_msms_dir
        )

        print(
            "Ion Mobility support:",
            has_ion_mobility
        )

        return (
            msms_dir,
            ccs_dir,
            rt_dir,
            candidate_msms_dir,
            candidate_rtccs_dir,
            has_ion_mobility
        )
