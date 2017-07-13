# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import tempfile
from .MetaMap import MetaMap
from .Concept import Corpus

class SubprocessBackend(MetaMap):
    def __init__(self, metamap_filename, version=None):
        """ Interface to MetaMap using subprocess. This creates a
            command line call to a specified metamap process.
        """
        MetaMap.__init__(self, metamap_filename, version)

    def extract_concepts(self, sentences=None, ids=None,
                         composite_phrase=4, filename=None,
                         file_format='sldi', allow_acronym_variants=False,
                         word_sense_disambiguation=False, allow_large_n=False,
                         no_derivational_variants=False,
                         derivational_variants=False, ignore_word_order=False,
                         unique_acronym_variants=False,
                         prefer_multiple_concepts=False,
                         ignore_stop_phrases=False, compute_all_mappings=False):
        """ extract_concepts takes a list of sentences and ids(optional)
            then returns a list of Concept objects extracted via
            MetaMap.

            Supported Options:
                Composite Phrase -Q
                Word Sense Disambiguation -y
                allow large N -l
                No Derivational Variants -d
                Derivational Variants -D
                Ignore Word Order -i
                Allow Acronym Variants -a
                Unique Acronym Variants -u
                Prefer Multiple Concepts -Y
                Ignore Stop Phrases -K
                Compute All Mappings -b

            For information about the available options visit
            http://metamap.nlm.nih.gov/.

            Note: If an error is encountered the process will be closed
                  and whatever was processed, if anything, will be
                  returned along with the error found.
        """
        if allow_acronym_variants and unique_acronym_variants:
            raise ValueError("You can't use both allow_acronym_variants and "
                             "unique_acronym_variants.")
        if (sentences is not None and filename is not None) or \
                (sentences is None and filename is None):
            raise ValueError("You must either pass a list of sentences "
                             "OR a filename.")
        if file_format not in ['sldi','sldiID']:
            raise ValueError("file_format must be either sldi or sldiID")

        input_file = None
        if sentences is not None:
            input_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        else:
            input_file = open(filename, 'r')
        output_file = tempfile.NamedTemporaryFile(mode="r", delete=False)
        error = None
        try:
            if sentences is not None:
                if ids is not None:
                    for identifier, sentence in zip(ids, sentences):
                        input_file.write('{0!r}|{1!r}\n'.format(identifier, sentence).encode('utf8'))
                else:
                    for sentence in sentences:
                        input_file.write('{0!r}\n'.format(sentence).encode('utf8'))
                input_file.flush()

            command = [self.metamap_filename, '-N']
            command.append('-Q')
            command.append(str(composite_phrase))
            if word_sense_disambiguation:
                command.append('-y')
            if allow_large_n:
                command.append('-l')
            if no_derivational_variants:
                command.append('-d')
            if derivational_variants:
                command.append('-D')
            if ignore_word_order:
                command.append('-i')
            if allow_acronym_variants:
                command.append('-a')
            if unique_acronym_variants:
                command.append('-u')
            if prefer_multiple_concepts:
                command.append('-Y')
            if ignore_stop_phrases:
                command.append('-K')
            if compute_all_mappings:
                command.append('-b')
            if ids is not None or (file_format == 'sldiID' and
                    sentences is None):
                command.append('--sldiID')
            else:
                command.append('--sldi')
            command.append(input_file.name)
            command.append(output_file.name)

            metamap_process = subprocess.Popen(command, stdout=subprocess.PIPE)
            while metamap_process.poll() is None:
                stdout = str(metamap_process.stdout.readline())
                if 'ERROR' in stdout:
                    metamap_process.terminate()
                    error = stdout.rstrip()
            output = str(output_file.read())
        finally:
            if sentences is not None:
                os.remove(input_file.name)
            else:
                input_file.close()
            os.remove(output_file.name)

        concepts = Corpus.load(output.splitlines())
        return (concepts, error)
