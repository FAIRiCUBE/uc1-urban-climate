from configparser import ConfigParser
import sqlalchemy as sa # conection to the database

def config(filename, section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))
    return db

def create_engine(db_config):
    keys = config(filename=db_config)
    return sa.create_engine(f"postgresql://{keys['user']}:{keys['password']}@{keys['host']}:{str(keys['port'])}/{keys['database']}")