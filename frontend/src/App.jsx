// import { useState } from 'react'
// // import reactLogo from './assets/react.svg'
// // import viteLogo from '/vite.svg'
// import './App.css'

// function App() {
//   const [count, setCount] = useState(0)

//   return (
//     <>
//       <div>
//         <a href="https://vite.dev" target="_blank">
//           <img src={viteLogo} className="logo" alt="Vite logo" />
//         </a>
//         <a href="https://react.dev" target="_blank">
//           <img src={reactLogo} className="logo react" alt="React logo" />
//         </a>
//       </div>
//       <h1>Vite + React</h1>
//       <div className="card">
//         <button onClick={() => setCount((count) => count + 1)}>
//           count is {count}
//         </button>
//         <p>
//           Edit <code>src/App.jsx</code> and save to test HMR
//         </p>
//       </div>
//       <p className="read-the-docs">
//         Click on the Vite and React logos to learn more
//       </p>
//     </>
//   )
// }

// export default App








import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const predict = async () => {
  if (!image) {
    alert("Please upload an image");
    return;
  }

  const formData = new FormData();
  formData.append("image", image);

  setLoading(true);
  setResult(null);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict/upload", {
      method: "POST",
      body: formData,
    });

    console.log("HTTP status:", response.status);

    const text = await response.text();   // 👈 IMPORTANT
    console.log("Raw response:", text);

    const data = JSON.parse(text);         // 👈 Parse manually
    console.log("Parsed JSON:", data);

    setResult(data);
  } catch (error) {
    console.error("Frontend error:", error);
    alert("Prediction failed");
  } finally {
    setLoading(false);
  }
};


  return (
    <div className="container">
      <h1>Retinal Eye Disease Detection</h1>
      <p>Upload a retinal image to detect CNV, DME, DRUSEN or NORMAL.</p>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setImage(e.target.files[0])}
      />

      <button onClick={predict}>Predict</button>

      {loading && <p className="loading">🔍 Analyzing image...</p>}

      {result && (
        <div className="result">
          <h2>Prediction Result</h2>
          <p>
            <strong>Best Model:</strong> {result.best_model}
          </p>

          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence (%)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.results).map(([model, data]) =>
                data.error ? null : (
                  <tr key={model}>
                    <td>{model}</td>
                    <td>{data.predicted_class}</td>
                    <td>{data.confidence}</td>
                  </tr>
                )
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
