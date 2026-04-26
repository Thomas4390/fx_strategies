//+------------------------------------------------------------------+
//| FxLogger.mqh                                                     |
//| Logging unifié vers Print + fichier CSV optionnel.               |
//+------------------------------------------------------------------+
#ifndef __FX_LOGGER_MQH__
#define __FX_LOGGER_MQH__

//+------------------------------------------------------------------+
//| Logger simple. Tag chaque ligne avec [SLEEVE] ou [GLOBAL].       |
//+------------------------------------------------------------------+
class CFxLogger
{
private:
    bool   m_verbose;
    int    m_file_handle;
    bool   m_to_file;
    string m_path;

public:
    CFxLogger() : m_verbose(false), m_file_handle(INVALID_HANDLE),
                  m_to_file(false), m_path("") {}

    void Init(bool verbose, bool to_file = false, string filename = "fx_log.csv")
    {
        m_verbose = verbose;
        m_to_file = to_file;
        m_path = filename;
        if(m_to_file)
        {
            m_file_handle = FileOpen(m_path, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
            if(m_file_handle == INVALID_HANDLE)
                PrintFormat("CFxLogger: cannot open %s (err=%d)",
                            m_path, GetLastError());
            else
                FileWrite(m_file_handle, "ts_utc", "tag", "level", "msg");
        }
    }

    void Shutdown()
    {
        if(m_file_handle != INVALID_HANDLE)
        {
            FileClose(m_file_handle);
            m_file_handle = INVALID_HANDLE;
        }
    }

    void Info(string tag, string msg)
    {
        Write("INFO", tag, msg);
    }

    void Warn(string tag, string msg)
    {
        Write("WARN", tag, msg);
    }

    void Error(string tag, string msg)
    {
        Write("ERROR", tag, msg);
    }

    void Debug(string tag, string msg)
    {
        if(!m_verbose) return;
        Write("DEBUG", tag, msg);
    }

private:
    void Write(string level, string tag, string msg)
    {
        string line = StringFormat("[%s][%s] %s", tag, level, msg);
        Print(line);
        if(m_file_handle != INVALID_HANDLE)
        {
            FileWrite(m_file_handle,
                      TimeToString(TimeGMT(), TIME_DATE | TIME_SECONDS),
                      tag, level, msg);
            FileFlush(m_file_handle);
        }
    }
};

//--- Logger global (instancié dans l'EA principal)
CFxLogger g_logger;

#endif // __FX_LOGGER_MQH__
