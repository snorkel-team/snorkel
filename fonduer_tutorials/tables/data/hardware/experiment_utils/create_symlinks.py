import os

DATA_PATH = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/'

train_digikey_html = DATA_PATH + "train_digikey/html/"
train_digikey_pdf = DATA_PATH + "train_digikey/pdf/"
train_other_html = DATA_PATH + "train_other/html/"
train_other_pdf = DATA_PATH + "train_other/pdf/"
dev_pdf = DATA_PATH + "dev/pdf/"
dev_html = DATA_PATH + "dev/html/"

# NOTE: You will need to create these folders beforehand. 
output_html = DATA_PATH + "symlinked_html/"
output_pdf = DATA_PATH + "symlinked_pdf/"

directories = [(dev_pdf, dev_html), (train_digikey_pdf, train_digikey_html), (train_other_pdf, train_other_html)]

def create_symlinks(num_symlinks):
    counter = 0
    while counter <= num_symlinks:
        for pdf_directory, html_directory in directories:
            src_files = os.listdir(pdf_directory)
            for pdf in src_files:
                if (pdf.startswith('.')):
                    continue
                if counter <= num_symlinks:
                    # Grab source HTML and PDF
                    pdf_filename = os.path.join(pdf_directory, pdf)
                    html_filename = os.path.join(html_directory, pdf[:-3] + "html")

                    # Generate corresponding symlink paths
                    counter += 1 # using this counter as the filename
                    sym_name = str(counter)
                    sym_pdf = os.path.join(output_pdf, sym_name + ".pdf")
                    sym_html = os.path.join(output_html, sym_name + ".html")

                    # Create the symlnks
                    if (os.path.isfile(pdf_filename) and os.path.isfile(html_filename)):
                        # Create symlinks for both the PDF and HTML
                        os.symlink(pdf_filename, sym_pdf)
                        os.symlink(html_filename, sym_html)
                else:
                    return

if __name__ == "__main__":
    # This determines how many symlinks to create.
    create_symlinks(1e6)
