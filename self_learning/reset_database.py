import sqlite3
import logging

# init logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Doomsday 
def reset_database():
    try:
        # Connect with database
        conn = sqlite3.connect('knowledge_base.db')
        c = conn.cursor()

        # Delete all contents in the database
        c.execute('DELETE FROM knowledge')
        conn.commit()


        # Optionally reset the primary key counter
        c.execute('DELETE FROM sqlite_sequence WHERE name="knowledge"')
        conn.commit()

        logging.info(f"Database reset successfully.")

    except Exception as e:
        logging.error(f"Error resetting the database: {e}")

    finally:
        # Close connection with database
        conn.close()

if __name__ == "__main__":
    reset_database()
    print("Database reset successfully.")

