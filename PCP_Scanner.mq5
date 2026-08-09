//+------------------------------------------------------------------+
//|                           PCP_Scanner.mq5                        |
//|         Put-Call Parity Arbitrage Scanner — MetaTrader 5         |
//|                                                                  |
//|  Methodology (mirrors Python backtest pcp_backtest.py):          |
//|  1. Enumerate all options in Market Watch                        |
//|  2. Group by (underlying, expiry)                                |
//|  3. For each group with >= MinPairs pairs:                       |
//|     a. Fit OLS: (C-P)_mid = a + b * K  across all strikes       |
//|     b. exec_edge = deviation of each pair from the OLS line,    |
//|        measured using actual bid/ask (not mid)                   |
//|  4. Display signals sorted by exec_edge in an on-chart panel     |
//|  5. Fire Alert() when edge exceeds AlertEdge threshold           |
//|                                                                  |
//|  Usage:                                                          |
//|  - Attach to ANY chart (e.g. EURUSD H1)                         |
//|  - Add BOVESPA option symbols to Market Watch first              |
//|  - Scanner runs on a timer — no ticks needed from main symbol    |
//+------------------------------------------------------------------+
#property copyright "Fernando Dorneles / PCP Backtest Project"
#property version   "1.10"
#property strict

//--- ─────────────────────────────────────────────────────────────────
//    Inputs
//--- ─────────────────────────────────────────────────────────────────
input group "=== Signal Filter ==="
input double InpMinEdge      = 0.05;   // Min exec_edge to display (option price units)
input double InpAlertEdge    = 0.50;   // Edge above this fires an Alert
input int    InpMinPairs     = 3;      // Min matched pairs per group to run OLS

input group "=== Scanner Timing ==="
input int    InpScanInterval = 10;     // Refresh every N seconds
input bool   InpRunOnTick    = false;  // Also refresh on every tick (heavier)

input group "=== Display ==="
input int    InpMaxRows      = 22;     // Max signal rows shown
input int    InpPanelX       = 10;     // Panel X offset (pixels from left)
input int    InpPanelY       = 28;     // Panel Y offset (pixels from top)

input group "=== Alerts ==="
input bool   InpAlertPopup   = true;   // Show pop-up alert
input bool   InpAlertPrint   = true;   // Print signal to Journal

//--- ─────────────────────────────────────────────────────────────────
//    Panel Layout Constants
//--- ─────────────────────────────────────────────────────────────────
#define P_NAME   "PCP_"       // prefix for all chart objects
#define ROW_H    17           // pixels per row
#define FONT_SZ  8
#define FONT     "Courier New"

// Column X offsets (relative to InpPanelX)
#define CX_C     0            // call symbol
#define CX_P     118          // put symbol
#define CX_K     236          // strike
#define CX_DTE   306          // DTE
#define CX_CB    348          // call bid
#define CX_CA    398          // call ask
#define CX_PB    448          // put bid
#define CX_PA    498          // put ask
#define CX_CPM   548          // C-P mid
#define CX_FIT   594          // OLS fit
#define CX_EDGE  640          // exec_edge + side
#define PANEL_W  710          // total width

//--- ─────────────────────────────────────────────────────────────────
//    Data Structures
//--- ─────────────────────────────────────────────────────────────────
struct OptionInfo {
    string   sym;
    int      right;        // 0=call, 1=put
    double   strike;
    datetime expiry;
    string   basis;        // underlying name
    double   bid, ask, mid;
};

struct Signal {
    string   c_sym, p_sym;
    string   basis;
    datetime expiry;
    double   strike;
    int      dte;
    double   c_bid, c_ask, c_mid;
    double   p_bid, p_ask, p_mid;
    double   cp_mid;       // c_mid - p_mid
    double   cp_fit;       // OLS fitted value
    double   exec_edge;
    string   side;         // "BUY(C-P)" or "SEL(C-P)"
};

//--- ─────────────────────────────────────────────────────────────────
//    Globals
//--- ─────────────────────────────────────────────────────────────────
Signal   g_sig[];
int      g_sig_n   = 0;
int      g_opts_n  = 0;    // total options scanned
int      g_grps_n  = 0;    // total (basis,expiry) groups
datetime g_last_scan = 0;
string   g_prev_alert_key = ""; // debounce alerts

//+------------------------------------------------------------------+
//  Init / Deinit / Tick / Timer
//+------------------------------------------------------------------+
int OnInit() {
    EventSetTimer(InpScanInterval);
    DeleteAllObjects();
    DoScan();
    DrawPanel();
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
    EventKillTimer();
    DeleteAllObjects();
}

void OnTick() {
    if(InpRunOnTick) { DoScan(); DrawPanel(); }
}

void OnTimer() {
    DoScan();
    DrawPanel();
}

//+------------------------------------------------------------------+
//  OLS  y = a + b*x
//+------------------------------------------------------------------+
void OLS(const double &x[], const double &y[], int n, double &a, double &b) {
    if(n < 2) { a = (n == 1) ? y[0] : 0.0; b = 0.0; return; }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for(int i = 0; i < n; i++) {
        sx  += x[i];
        sy  += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    double d = n * sxx - sx * sx;
    if(MathAbs(d) < 1e-12) { a = sy / n; b = 0.0; return; }
    b = (n * sxy - sx * sy) / d;
    a = (sy - b * sx) / n;
}

//+------------------------------------------------------------------+
//  Infer option right (call/put) from B3 naming convention
//  B3 equity options:  <TICKER> + <month_letter> + <strike_digits>
//  Call month letters: A(Jan) B(Feb) C(Mar) D(Apr) E(May) F(Jun)
//                      G(Jul) H(Aug) I(Sep) J(Oct) K(Nov) L(Dec)
//  Put  month letters: M(Jan) N(Feb) O(Mar) P(Apr) Q(May) R(Jun)
//                      S(Jul) T(Aug) U(Sep) V(Oct) W(Nov) X(Dec)
//+------------------------------------------------------------------+
int GuessRight(const string sym, const string basis) {
    long r;
    if(SymbolInfoInteger(sym, SYMBOL_OPTION_RIGHT, r)) {
        if(r == SYMBOL_OPTION_RIGHT_CALL) return 0;
        if(r == SYMBOL_OPTION_RIGHT_PUT)  return 1;
    }
    // Fallback: month letter position = len(basis)
    int blen = StringLen(basis);
    if(StringLen(sym) > blen) {
        ushort ch = StringGetCharacter(sym, blen);
        if(ch >= 'A' && ch <= 'L') return 0;
        if(ch >= 'M' && ch <= 'X') return 1;
    }
    return -1;
}

//+------------------------------------------------------------------+
//  Extract underlying name from symbol
//  Strategy:
//   1. Use SYMBOL_BASIS if set by broker
//   2. Walk the symbol name: find first letter in [A-X] immediately
//      followed by a digit — that is the month code boundary
//+------------------------------------------------------------------+
string GetBasis(const string sym) {
    string b;
    if(SymbolInfoString(sym, SYMBOL_BASIS, b) && StringLen(b) > 0) return b;

    int n = StringLen(sym);
    for(int i = 1; i < n - 1; i++) {
        ushort c    = StringGetCharacter(sym, i);
        ushort nxt  = StringGetCharacter(sym, i + 1);
        bool is_month  = (c >= 'A' && c <= 'X');
        bool nxt_digit = (nxt >= '0' && nxt <= '9');
        if(is_month && nxt_digit) return StringSubstr(sym, 0, i);
    }
    return StringSubstr(sym, 0, 4); // fallback
}

//+------------------------------------------------------------------+
//  Main Scan
//+------------------------------------------------------------------+
void DoScan() {
    datetime now = TimeCurrent();
    g_sig_n  = 0;
    g_opts_n = 0;
    g_grps_n = 0;
    ArrayResize(g_sig, 0);

    // ── Step 1: collect all options from Market Watch ──────────────
    int total = SymbolsTotal(true);
    OptionInfo opts[];
    int n_opts = 0;

    for(int i = 0; i < total; i++) {
        string sym = SymbolName(i, true);

        // Must have a positive strike to be an option
        double strike;
        if(!SymbolInfoDouble(sym, SYMBOL_OPTION_STRIKE, strike)) continue;
        if(strike <= 0.0) continue;

        datetime expiry = (datetime)SymbolInfoInteger(sym, SYMBOL_EXPIRATION_TIME);
        if(expiry <= 0) continue;
        if(expiry < now) continue;  // already expired

        string basis = GetBasis(sym);
        int right = GuessRight(sym, basis);
        if(right < 0) continue;  // couldn't determine call/put

        double bid = SymbolInfoDouble(sym, SYMBOL_BID);
        double ask = SymbolInfoDouble(sym, SYMBOL_ASK);
        // Require at least one side quoted
        if(bid <= 0.0 && ask <= 0.0) continue;

        double mid;
        if(bid > 0.0 && ask > 0.0) mid = (bid + ask) / 2.0;
        else if(ask > 0.0)          mid = ask;
        else                         mid = bid;

        ArrayResize(opts, n_opts + 1);
        opts[n_opts].sym    = sym;
        opts[n_opts].right  = right;
        opts[n_opts].strike = strike;
        opts[n_opts].expiry = expiry;
        opts[n_opts].basis  = basis;
        opts[n_opts].bid    = bid;
        opts[n_opts].ask    = ask;
        opts[n_opts].mid    = mid;
        n_opts++;
    }

    g_opts_n = n_opts;
    if(n_opts == 0) {
        if(InpAlertPrint) Print("PCP_Scanner: no options in Market Watch — add BOVESPA/OPCOES symbols first");
        g_last_scan = now;
        return;
    }

    // ── Step 2: build unique group keys = basis + "_" + expiry_int ─
    string grp_keys[];
    int n_grps = 0;
    for(int i = 0; i < n_opts; i++) {
        string key = opts[i].basis + "_" + IntegerToString((long)opts[i].expiry);
        bool found = false;
        for(int g = 0; g < n_grps; g++) if(grp_keys[g] == key) { found = true; break; }
        if(!found) { ArrayResize(grp_keys, n_grps + 1); grp_keys[n_grps++] = key; }
    }
    g_grps_n = n_grps;

    // ── Step 3: for each group, match pairs and run OLS ────────────
    for(int g = 0; g < n_grps; g++) {
        string key = grp_keys[g];

        OptionInfo calls[], puts_arr[];
        int nc = 0, np = 0;
        for(int i = 0; i < n_opts; i++) {
            string k = opts[i].basis + "_" + IntegerToString((long)opts[i].expiry);
            if(k != key) continue;
            if(opts[i].right == 0) { ArrayResize(calls,    nc + 1); calls[nc++]    = opts[i]; }
            else                   { ArrayResize(puts_arr, np + 1); puts_arr[np++] = opts[i]; }
        }

        // Match by strike (tolerance 0.01)
        OptionInfo pc[], pp[];
        int npairs = 0;
        for(int ci = 0; ci < nc; ci++) {
            for(int pi = 0; pi < np; pi++) {
                if(MathAbs(calls[ci].strike - puts_arr[pi].strike) < 0.01) {
                    ArrayResize(pc, npairs + 1); pc[npairs] = calls[ci];
                    ArrayResize(pp, npairs + 1); pp[npairs] = puts_arr[pi];
                    npairs++;
                    break;
                }
            }
        }
        if(npairs < InpMinPairs) continue;

        // OLS on mid: cp_mid = a + b * K
        double xv[], yv[];
        ArrayResize(xv, npairs); ArrayResize(yv, npairs);
        for(int i = 0; i < npairs; i++) {
            xv[i] = pc[i].strike;
            yv[i] = pc[i].mid - pp[i].mid;
        }
        double ols_a, ols_b;
        OLS(xv, yv, npairs, ols_a, ols_b);

        // Compute exec_edge per pair
        for(int i = 0; i < npairs; i++) {
            double c_bid = pc[i].bid, c_ask = pc[i].ask;
            double p_bid = pp[i].bid, p_ask = pp[i].ask;
            double cp_fit = ols_a + ols_b * pc[i].strike;
            double cp_mid = pc[i].mid - pp[i].mid;

            // buy(C-P):  pay c_ask − receive p_bid; profit if cp_fit > cost
            double buy_edge  = cp_fit - (c_ask - p_bid);
            // sell(C-P): receive c_bid − pay p_ask; profit if revenue > cp_fit
            double sell_edge = (c_bid - p_ask) - cp_fit;

            double edge = 0; string side = "";
            if(buy_edge  >= sell_edge && buy_edge  > InpMinEdge) { edge = buy_edge;  side = "BUY(C-P)"; }
            else if(sell_edge > buy_edge && sell_edge > InpMinEdge) { edge = sell_edge; side = "SEL(C-P)"; }
            if(edge <= InpMinEdge) continue;

            int dte = (int)((pc[i].expiry - now) / 86400);
            if(dte < 0) continue;

            ArrayResize(g_sig, g_sig_n + 1);
            g_sig[g_sig_n].c_sym     = pc[i].sym;
            g_sig[g_sig_n].p_sym     = pp[i].sym;
            g_sig[g_sig_n].basis     = pc[i].basis;
            g_sig[g_sig_n].expiry    = pc[i].expiry;
            g_sig[g_sig_n].strike    = pc[i].strike;
            g_sig[g_sig_n].dte       = dte;
            g_sig[g_sig_n].c_bid     = c_bid;
            g_sig[g_sig_n].c_ask     = c_ask;
            g_sig[g_sig_n].c_mid     = pc[i].mid;
            g_sig[g_sig_n].p_bid     = p_bid;
            g_sig[g_sig_n].p_ask     = p_ask;
            g_sig[g_sig_n].p_mid     = pp[i].mid;
            g_sig[g_sig_n].cp_mid    = cp_mid;
            g_sig[g_sig_n].cp_fit    = cp_fit;
            g_sig[g_sig_n].exec_edge = edge;
            g_sig[g_sig_n].side      = side;
            g_sig_n++;
        }
    }

    // Sort by exec_edge descending
    for(int i = 0; i < g_sig_n - 1; i++)
        for(int j = i + 1; j < g_sig_n; j++)
            if(g_sig[j].exec_edge > g_sig[i].exec_edge) {
                Signal tmp = g_sig[i]; g_sig[i] = g_sig[j]; g_sig[j] = tmp;
            }

    // Alerts for top signal
    if(g_sig_n > 0 && g_sig[0].exec_edge >= InpAlertEdge) {
        string alert_key = g_sig[0].c_sym + "_" + g_sig[0].p_sym
                           + "_" + DoubleToString(g_sig[0].exec_edge, 3);
        if(alert_key != g_prev_alert_key) {
            g_prev_alert_key = alert_key;
            string msg = StringFormat("PCP SIGNAL  %s / %s  K=%.0f  DTE=%d  edge=%.4f  [%s]",
                g_sig[0].c_sym, g_sig[0].p_sym,
                g_sig[0].strike, g_sig[0].dte,
                g_sig[0].exec_edge, g_sig[0].side);
            if(InpAlertPopup) Alert(msg);
            if(InpAlertPrint) Print(msg);
        }
    }

    if(InpAlertPrint)
        Print(StringFormat("PCP_Scanner: %d signals | %d options | %d groups | %s",
            g_sig_n, g_opts_n, g_grps_n,
            TimeToString(now, TIME_DATE | TIME_MINUTES)));

    g_last_scan = now;
}

//+------------------------------------------------------------------+
//  Panel Drawing Helpers
//+------------------------------------------------------------------+
void DeleteAllObjects() {
    int total = ObjectsTotal(0, 0, -1);
    for(int i = total - 1; i >= 0; i--) {
        string name = ObjectName(0, i, 0, -1);
        if(StringFind(name, P_NAME) == 0) ObjectDelete(0, name);
    }
}

void MakeRect(const string name, int x, int y, int w, int h, color bg) {
    if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE,   x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE,   y);
    ObjectSetInteger(0, name, OBJPROP_XSIZE,       w);
    ObjectSetInteger(0, name, OBJPROP_YSIZE,       h);
    ObjectSetInteger(0, name, OBJPROP_BGCOLOR,     bg);
    ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(0, name, OBJPROP_COLOR,       bg);
    ObjectSetInteger(0, name, OBJPROP_CORNER,      CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_BACK,        true);
}

void MakeText(const string name, int x, int y, const string txt, color fg, int sz = FONT_SZ) {
    if(ObjectFind(0, name) < 0) ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
    ObjectSetInteger(0, name, OBJPROP_XDISTANCE,  x);
    ObjectSetInteger(0, name, OBJPROP_YDISTANCE,  y + 2);
    ObjectSetInteger(0, name, OBJPROP_COLOR,      fg);
    ObjectSetString( 0, name, OBJPROP_TEXT,       txt);
    ObjectSetString( 0, name, OBJPROP_FONT,       FONT);
    ObjectSetInteger(0, name, OBJPROP_FONTSIZE,   sz);
    ObjectSetInteger(0, name, OBJPROP_CORNER,     CORNER_LEFT_UPPER);
    ObjectSetInteger(0, name, OBJPROP_BACK,       false);
}

void Cell(const string id, int col_x, int row_y, int w, const string txt,
          color fg, color bg, int sz = FONT_SZ) {
    int ax = InpPanelX + col_x;
    MakeRect(P_NAME + id + "_bg", ax, row_y, w, ROW_H, bg);
    MakeText(P_NAME + id,         ax + 2, row_y, txt, fg, sz);
}

color EdgeColor(double e) {
    if(e >= 1.00) return clrLime;
    if(e >= 0.50) return clrYellow;
    if(e >= 0.20) return clrOrange;
    return clrSilver;
}

//+------------------------------------------------------------------+
//  Full Panel Render
//+------------------------------------------------------------------+
void DrawPanel() {
    int px = InpPanelX;
    int py = InpPanelY;

    // ── Title bar ─────────────────────────────────────────────────
    string title = StringFormat(" PCP SCANNER   %s   Sigs: %d  |  Opts: %d  |  Groups: %d",
        TimeToString(g_last_scan, TIME_DATE | TIME_MINUTES),
        g_sig_n, g_opts_n, g_grps_n);
    MakeRect(P_NAME+"title_bg", px, py, PANEL_W, ROW_H, C'10,10,50');
    MakeText(P_NAME+"title",    px + 3, py, title, clrWhite, FONT_SZ);

    // ── Header row ────────────────────────────────────────────────
    int hy = py + ROW_H + 1;
    color hbg = C'30,30,70'; color hfg = C'160,160,220';
    Cell("hC",   CX_C,   hy, 115, "CALL",    hfg, hbg);
    Cell("hP",   CX_P,   hy, 115, "PUT",     hfg, hbg);
    Cell("hK",   CX_K,   hy,  65, "STRIKE",  hfg, hbg);
    Cell("hD",   CX_DTE, hy,  40, "DTE",     hfg, hbg);
    Cell("hCB",  CX_CB,  hy,  47, "C.Bid",   hfg, hbg);
    Cell("hCA",  CX_CA,  hy,  47, "C.Ask",   hfg, hbg);
    Cell("hPB",  CX_PB,  hy,  47, "P.Bid",   hfg, hbg);
    Cell("hPA",  CX_PA,  hy,  47, "P.Ask",   hfg, hbg);
    Cell("hCPM", CX_CPM, hy,  43, "C-P",     hfg, hbg);
    Cell("hFIT", CX_FIT, hy,  43, "FIT",     hfg, hbg);
    Cell("hEDG", CX_EDGE,hy,  67, "EDGE",    hfg, hbg);

    // ── Data rows ─────────────────────────────────────────────────
    int rows = MathMin(g_sig_n, InpMaxRows);
    for(int r = 0; r < rows; r++) {
        Signal s  = g_sig[r];
        int    ry = py + (r + 2) * (ROW_H + 1) + 1;
        color  bg = (r % 2 == 0) ? C'12,12,30' : C'18,18,42';
        color  ec = EdgeColor(s.exec_edge);
        string id = "r" + IntegerToString(r);

        Cell(id+"C",   CX_C,   ry, 115, s.c_sym,                               clrAqua,    bg);
        Cell(id+"P",   CX_P,   ry, 115, s.p_sym,                               clrViolet,  bg);
        Cell(id+"K",   CX_K,   ry,  65, DoubleToString(s.strike, 0),            clrWhite,   bg);
        Cell(id+"D",   CX_DTE, ry,  40, IntegerToString(s.dte)+"d",             clrGray,    bg);
        Cell(id+"CB",  CX_CB,  ry,  47, DoubleToString(s.c_bid, 2),             C'120,120,120', bg);
        Cell(id+"CA",  CX_CA,  ry,  47, DoubleToString(s.c_ask, 2),             C'120,120,120', bg);
        Cell(id+"PB",  CX_PB,  ry,  47, DoubleToString(s.p_bid, 2),             C'120,120,120', bg);
        Cell(id+"PA",  CX_PA,  ry,  47, DoubleToString(s.p_ask, 2),             C'120,120,120', bg);
        Cell(id+"CPM", CX_CPM, ry,  43, DoubleToString(s.cp_mid, 2),            clrYellow,  bg);
        Cell(id+"FIT", CX_FIT, ry,  43, DoubleToString(s.cp_fit, 2),            clrYellow,  bg);
        Cell(id+"EDG", CX_EDGE,ry,  67, s.side+" "+DoubleToString(s.exec_edge,3), ec, bg);
    }

    // ── Erase rows beyond current count ──────────────────────────
    for(int r = rows; r < InpMaxRows; r++) {
        string id = "r" + IntegerToString(r);
        string cols[] = {"C","P","K","D","CB","CA","PB","PA","CPM","FIT","EDG"};
        for(int c = 0; c < ArraySize(cols); c++) {
            string n = P_NAME + id + cols[c];
            if(ObjectFind(0, n)       >= 0) ObjectDelete(0, n);
            if(ObjectFind(0, n+"_bg") >= 0) ObjectDelete(0, n+"_bg");
        }
    }

    // ── Footer: no-signal message ─────────────────────────────────
    if(g_sig_n == 0) {
        int fy = py + 2 * (ROW_H + 1) + 1;
        MakeRect(P_NAME+"nomsg_bg", px, fy, PANEL_W, ROW_H * 2, C'12,12,30');
        string msg = (g_opts_n == 0)
            ? "No options in Market Watch. Add BOVESPA/OPCOES symbols."
            : StringFormat("No signals >= %.2f found across %d options (%d groups). Lower MinEdge?",
                           InpMinEdge, g_opts_n, g_grps_n);
        MakeText(P_NAME+"nomsg", px + 6, fy + 2, msg, clrGray);
    } else {
        if(ObjectFind(0, P_NAME+"nomsg_bg") >= 0) ObjectDelete(0, P_NAME+"nomsg_bg");
        if(ObjectFind(0, P_NAME+"nomsg")    >= 0) ObjectDelete(0, P_NAME+"nomsg");
    }

    ChartRedraw(0);
}
//+------------------------------------------------------------------+
