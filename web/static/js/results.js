'use strict';

class Header extends React.Component {
  constructor(props) {
    super(props);
    this.create_query_input = this.create_query_input.bind(this);
    this.create_tags = this.create_tags.bind(this);
  }

  create_query_input() {
    switch(this.props.data.input_modality) {
      case "text":
        return this.props.data.query_input;
      case "image":
        return "<img height='100' src="+this.props.data.query_input+"/>";
    }  
  }

  create_tags() {
    var tags = "";
    for (var i = 0; i < this.props.data.num_datasets; i++) {
      var result = this.props.data.results[i];
      if (result.modality == "text") {
        for (var j = 0; j < result.num_results; j++) {
          tags = tags + result.data[j] + ", "
        }
      }
    }
    return tags;
  }

  render() {
    console.log("Rendering header...");
    var input_modality = this.props.data.input_modality;
    var query_input = this.create_query_input();
    var tags = this.create_tags();
    return (
      <div id="header">
        <div className="row justify-content-center">
          <div className="col-8 text-center">
            <h1 className="pt-5">Query:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
          <div className="col-1 text-center">
            <h4>Modality:</h4> {input_modality}<br/>
            <a className="btn btn-primary pt-3" role="button" href="/">Search Another</a>
          </div>
          <div className="col-4 text-center">
            <h4>Input:</h4>
            <div dangerouslySetInnerHTML={{__html: query_input}}></div>
          </div>
          <div className="col-4 text-center">
            <h4>Relevant Tags:</h4>
            {tags}
          </div>
        </div>
      </div>
    )
  }
}

class Content extends React.Component {
  render() {
    var data = JSON.parse(this.props.data);
    console.log(data);
    return (
      <Header data={data}></Header>
    );
  }
}
