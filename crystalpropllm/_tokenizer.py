import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines()]


ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
         "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
         "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
         "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
         "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"]

DIGITS = [str(d) for d in list(range(10))]

KEYWORDS = [
    "_cell_length_b",
    "_atom_site_occupancy",
    "_atom_site_attached_hydrogens",
    "_cell_length_a",
    "_cell_angle_beta",
    "_symmetry_equiv_pos_as_xyz",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_structural",
    "_chemical_name_systematic",
    "_atom_site_fract_y",
    "_atom_site_symmetry_multiplicity",
    "_chemical_formula_sum",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_cell_length_c",
    "_atom_site_B_iso_or_equiv",
    "_symmetry_equiv_pos_site_id",
    "_cell_volume",
    "_atom_site_fract_z",
    "_cell_angle_alpha",
    "_cell_formula_units_Z",
    "loop_",
    "data_"
]

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",  # does not exist in CIF spec
    "_atom_type_radius",  # does not exist in CIF spec
    "_atom_type_ionic_radius",  # does not exist in CIF spec
    "_atom_type_oxidation_number",
    # Block identifiers - these are always needed
    "FEATURE_",
    "OPERATION_", 
    "TARGET_"
]

UNK_TOKEN = "<unk>"


class CIFTokenizer:
    def __init__(self, vocab_file=None):
        """
        Initialize CIFTokenizer with optional vocabulary file.
        
        Parameters
        ----------
        vocab_file : str, optional
            Path to vocabulary file containing additional tokens (one per line)
        """
        self._tokens = list(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())
        
        # Load additional vocabulary from file if provided
        if vocab_file and os.path.exists(vocab_file):
            additional_tokens = self.load_vocab_file(vocab_file)
            self._tokens.extend(additional_tokens)
            print(f"Loaded {len(additional_tokens)} additional tokens from {vocab_file}")

        space_groups = list(self.space_groups())
        # Replace 'Pm' space group with 'Pm_sg' to disambiguate from atom 'Pm',
        #  or 'P1' with 'P1_sg' to disambiguate from atom 'P' and number '1'
        space_groups_sg = [sg+"_sg" for sg in space_groups]
        self._tokens.extend(space_groups_sg)

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        self._tokens_with_unk = list(self._tokens)
        self._tokens_with_unk.append(UNK_TOKEN)

        # a mapping from characters to integers
        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens_with_unk)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens_with_unk)}
        # map the id of 'Pm_sg' back to 'Pm', or 'P1_sg' to 'P1',
        #  for decoding convenience
        for sg in space_groups_sg:
            self._id_to_token[self.token_to_id[sg]] = sg.replace("_sg", "")
    
    @staticmethod
    def load_vocab_file(vocab_file):
        """
        Load additional vocabulary tokens from a file.
        
        Parameters
        ----------
        vocab_file : str
            Path to vocabulary file (one token per line)
            
        Returns
        -------
        list
            List of additional tokens
        """
        additional_tokens = []
        try:
            with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
                for line in vocab_f:
                    token = line.strip()
                    if token and not token.startswith('#'):  # Skip empty lines and comments
                        additional_tokens.append(token)
        except (IOError, FileNotFoundError) as e:
            print(f"Warning: Could not load vocabulary file {vocab_file}: {e}")
        
        return additional_tokens

    @staticmethod
    def atoms():
        return ATOMS

    @staticmethod
    def digits():
        return DIGITS

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        kws.extend(EXTENDED_KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        # encoder: take a list of tokens, output a list of integers
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        # decoder: take a list of integers (i.e. encoded tokens), output a string
        return ''.join([self._id_to_token[i] for i in ids])

    def tokenize_cif(self, cif_string, single_spaces=True):
        # Preprocessing step to replace '_symmetry_space_group_name_H-M Pm'
        #  with '_symmetry_space_group_name_H-M Pm_sg',to disambiguate from atom 'Pm',
        #  or any space group symbol to avoid problematic cases, like 'P1'
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(fr'(_symmetry_space_group_name_H-M *\b({spacegroups}))\n', r'\1_sg\n', cif_string)

        # Create a regex pattern by joining the escaped tokens with '|'
        token_pattern = '|'.join(self._escaped_tokens)

        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'

        # Tokenize the input string using the regex pattern
        if single_spaces:
            cif_string = re.sub(r'[ \t]+', ' ', cif_string)
        tokens = re.findall(full_pattern, cif_string)

        # Replace unrecognized tokens with the unknown_token
        tokens = [token if token in self._tokens else UNK_TOKEN for token in tokens]

        # Generate block IDs with original numbering but detect structure start with data_
        block_ids = []
        current_block = 0  # structure (default)
        for token in tokens:
            if token == 'FEATURE_':
                current_block = 1  # material properties
            elif token == 'OPERATION_':
                current_block = 2  # operational conditions
            elif token == 'data_':
                current_block = 0  # structure block (starts with data_)
            elif token == 'TARGET_':
                current_block = 3  # target
            block_ids.append(current_block)

        return tokens, block_ids
