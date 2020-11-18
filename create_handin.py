import argparse
import zipfile
import subprocess
import pathlib


class TeamMember:
    def __init__(self, first_names, last_names):
        self.first_names = first_names
        self.last_names = last_names

    def __str__(self):
        return "-".join(self.last_names + self.first_names)


def main():
    # members with first and last names, must be sorted by last names, then first names
    members = sorted([
        TeamMember(["Ergin"], ["Kohen", "Sagner"]),
        TeamMember(["Nicolas"], ["Wolf"]),
        TeamMember(["Lars", "Erik"], ["Kühmichel"]),
    ], key=lambda t: "".join(t.last_names + t.first_names))

    # members string where individual members are separated by _
    members = "_".join([str(m) for m in members])

    parser = argparse.ArgumentParser(description="Automatically create the hand-in file for the lecture FML.")
    parser.add_argument("name", type=str,
                        help="The name of the exercise, to be included at the end of the zip filename.")
    parser.add_argument("notebook", type=str, help="The name of the Jupyter Notebook")
    parser.add_argument("-m", "--more", type=str,
                        help="Glob Pattern to determine additional files which should be added to the Hand-In.")

    args = parser.parse_args()

    # check if directory and file exist
    ex_path = pathlib.Path(args.name)
    if not ex_path.is_dir():
        raise ValueError(f"Exercise Path '{args.name}' does not exist.")
    nb_file = ex_path / (args.notebook + ".ipynb")

    if not nb_file.is_file():
        raise ValueError(f"Notebook '{args.notebook}.ipynb' does not exist in '{args.name}'.")

    # both exist, create an html file
    subprocess.run(f"jupyter nbconvert --to html {nb_file.absolute()}")

    html_file = ex_path / (args.notebook + ".html")

    zip_path = ex_path / (members + "_" + args.name + ".zip")

    # create the zip hand-in file
    zipf = zipfile.ZipFile(zip_path, "w")

    # add the html and notebook to the zip
    zipf.write(html_file, html_file.name)
    zipf.write(nb_file, nb_file.name)

    if args.more:
        files = list(ex_path.glob(args.more))
        print(f"adding {len(files)} more files...")

        for file in ex_path.glob(args.more):
            # add the file without adding the ex_path folder in
            zipf.write(file, pathlib.Path(*file.parts[1:]))

    zipf.close()

    return 0


if __name__ == "__main__":
    main()
