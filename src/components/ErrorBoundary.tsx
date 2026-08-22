import React from "react";

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
  info: string | null;
}

/**
 * Without this, a throw anywhere in the tree makes React 19 unmount the whole
 * app and the window goes blank with no clue what happened — indistinguishable
 * from a webview crash. Show the failure instead, and keep it copyable.
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null, info: null };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    this.setState({ info: info.componentStack ?? null });
    console.error("Unhandled render error", error, info);
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="crash-screen">
        <div className="crash-card">
          <h1>SmolTalk hit an error</h1>
          <p className="crash-sub">
            The interface stopped rather than showing a blank window. Your library is on disk and
            is unaffected — reloading is safe.
          </p>
          <pre className="crash-trace">
            {error.message}
            {error.stack ? `\n\n${error.stack}` : ""}
            {info ? `\n\nComponent stack:${info}` : ""}
          </pre>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    );
  }
}
