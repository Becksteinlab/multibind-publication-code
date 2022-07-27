from rdkit import Chem
from rdkit.Chem import Draw


def main():
    mol = Chem.MolFromSmiles('C(CN(CC(=O)[O-])CC(=O)[O-])N(CCN(CC(=O)[O-])CC(=O)[O-])CC(=O)[O-]')
    Draw.MolToFile(mol, "dtpa.png", size=(600, 600))

if __name__ == "__main__":
    main()
