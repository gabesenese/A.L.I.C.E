import os
from tqdm import tqdm
from convokit import Corpus, download

class DatasetLoader:
    def __init__(self):
        self.questions, self.answers = self.load_datasets()

    def load_datasets(self):
        questions = []
        answers = []
        dataset_path = "convokit_corpus/wiki-corpus"

        # Check if dataset file already exists
        if os.path.exists(dataset_path):
            print("Loading dataset from local directory ...")
            wiki_talk = Corpus(filename=dataset_path)
        else:
            print("Downloading and loading Wikipedia Talk dataset... ")
            wiki_talk = Corpus(download('wiki-corpus'))

        # Debugging
        print("\nSample Utterance Metadata: ")
        for i, utt in enumerate(wiki_talk.iter_utterances()):
            if i < 5:
                print(f"Utterance ID: {utt.id}, Speaker: {utt.speaker}, Reply_to: {utt.reply_to}, Text: {utt.text}")

        print("\nExtracting questions and answers from the dataset ... ")
        utterances = list(wiki_talk.iter_utterances())
        
        for utt in tqdm(utterances, desc="Processing utterances", unit="utterance"):
            reply_to_id = utt.reply_to
            if reply_to_id:
                reply_utt = wiki_talk.get_utterance(reply_to_id)
                if reply_utt:
                    questions.append(utt.text)
                    answers.append(reply_utt.text)
                else:
                    # Debug: print missing reply_to info
                    print(f"No reply_to utterance found for Utterance ID: {utt.id}")
                    #answers.append('')  # Handle case where reply_to utterance not found
            else:
                # Debug: print if reply_to is None
                print(f"Utterance ID {utt.id} has no reply_to (None)")
                answers.append('')  # Handle case where reply_to is None

        # Filter out cases where there is no valid answer
        print("\nFiltering valid questions-answer pairs ... ")
        valid_pairs = [(q, a) for q, a in zip(questions, answers) if a]
        questions, answers = zip(*valid_pairs) if valid_pairs else ([], [])

        return questions, answers

if __name__ == "__main__":

    print(f"Initializing Datasetloader ... ")
    dataset_loader = DatasetLoader()

    num_questions =  len(dataset_loader.questions)
    num_answers = len(dataset_loader.answers)
    
    print(f"\nNumber of questions: {num_questions}")
    print(f"\nNumber of answers: {num_answers}")

    sample_size = 5
    print(f"\nSample of questions (first {sample_size}): ")
    for question in dataset_loader.questions[:sample_size]:
        print(f"- {question}")

    print(f"\nSample of answers (first {sample_size}):")
    for answer in dataset_loader.answers[:sample_size]:
        print(f"- {answer}")

