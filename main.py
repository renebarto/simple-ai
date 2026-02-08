import argparse
import sys
from logging_utils import setup_logging
from model import Model
from ui import UI
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(description='MTBF Analysis Tool')
    
    # Required arguments
    parser.add_argument('command', choices=['build_model', 'chatbot'], 
                       help='Command to execute')
    parser.add_argument('--model', help='Model file', default='modelcsv')
    parser.add_argument('--log-dir', help='Directory for log files (default: logs)', default='logs')
    parser.add_argument('--content-dir', help='Directory for content to be used to create model (default: logs)', default='content')
    
    args = parser.parse_args()

    # Setup centralized logging
    logger = setup_logging('baremetal-ai.log', 'baremetal-ai', args.log_dir)
    logger.info("RAG based model creation / usage")
    logger.info(f"Command: {args.command}")
    logger.info(f"Log directory: {args.log_dir}")

    success = False

    try:
        if args.command == 'build_model':
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama')

            model = Model(client, logger)
            if not model.ingest(args.content_dir):
                print("ERROR: Ingestion failed")
                sys.exit(1)
            model.save(args.model)
            success = True
            
        elif args.command == 'chatbot':
            client = OpenAI(
                base_url = 'http://localhost:11434/v1',
                api_key='ollama')

            model = Model(client, logger)
            model.load(args.model)
            ui = UI()
            done = False
            while not done:
                print('Ask me a question:')
                prompt = ui.get_message()
                if prompt == '/bye':
                    done = True
                else:
                    response = model.query(prompt)
                    print(response.choices[0].message.content)

            success = True
    
    except Exception as e:
        logger.error(f"Operation failed with exception: {e}", exc_info=True)
        success = False

    if success:
        logger.info("Operation completed successfully")
    else:
        logger.error("Operation failed")
        sys.exit(1)


if __name__ == "__main__":
  main()