'use strict';

class Search extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: JSON.parse(this.props.data),
    };
  }

  render() {
      return (
        <div className="row justify-content-center w-100">
          <div className="col-8 text-center">
            <h1 className="pt-5">Search Engine</h1>
          </div>
        </div>

        <div className>
          
        </div>
        

      )
    }
  }
  